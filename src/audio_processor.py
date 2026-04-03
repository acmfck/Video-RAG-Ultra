import hashlib
import json
import os
import subprocess
from threading import Lock

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import whisper as openai_whisper
except ImportError:
    openai_whisper = None

try:
    from device_utils import resolve_device
except ImportError:
    from src.device_utils import resolve_device


DEFAULT_AUDIO_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "embeddings", "audio_cache")
)


class AudioRetriever:
    _shared_lock = Lock()
    _shared_asr_models = {}
    _shared_text_encoders = {}

    def __init__(
        self,
        whisper_model_size="large-v3",
        use_fp16=True,
        use_fast_index=False,
        chunk_seconds=300,
        cache_dir=DEFAULT_AUDIO_CACHE_DIR,
        runtime_dir="audio_runtime",
        asr_backend=None,
        text_model_name=None,
        use_vad=None,
    ):
        """
        Initialize the audio retriever with configurable ASR and text embedding backends.

        Defaults:
            - ASR backend: faster-whisper
            - ASR model: large-v3
            - VAD: enabled
            - Text embeddings: BAAI/bge-m3
        """
        self.device = resolve_device(preferred_cuda_indices=[2, 0], fallback="cpu")
        print(f"[Audio Init] Loading models on {self.device} (Total GPUs: {torch.cuda.device_count()})...")

        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.chunk_seconds = chunk_seconds
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.runtime_dir = runtime_dir
        os.makedirs(self.runtime_dir, exist_ok=True)
        self.use_fast_index = use_fast_index

        requested_backend = (asr_backend or os.getenv("AUDIO_ASR_BACKEND", "faster-whisper")).strip().lower()
        self.asr_model_name = whisper_model_size or os.getenv("AUDIO_ASR_MODEL", "large-v3")
        self.use_vad = (
            use_vad
            if use_vad is not None
            else os.getenv("AUDIO_ASR_USE_VAD", "1") != "0"
        )
        self.text_model_name = text_model_name or os.getenv("AUDIO_TEXT_EMBEDDING_MODEL", "BAAI/bge-m3")

        self.asr_backend, self.asr_model = self._init_asr_backend(requested_backend)
        self.text_encoder = self._load_text_encoder(self.text_model_name, self.device)

        self.dimension = self._get_text_embedding_dimension(self.text_encoder)
        self.index = self._build_index(use_fast_index)
        self.metadata = {}

    def _init_asr_backend(self, requested_backend):
        try:
            model = self._load_asr_model(
                backend_name=requested_backend,
                model_name=self.asr_model_name,
                device=self.device,
                use_fp16=self.use_fp16,
            )
            return requested_backend, model
        except RuntimeError as exc:
            if requested_backend == "whisper":
                raise

            fallback_backend = "whisper"
            fallback_model = os.getenv("AUDIO_ASR_FALLBACK_MODEL", self.asr_model_name or "medium")
            print(
                f"[Audio Warning] faster-whisper 不可用，回退到 OpenAI Whisper "
                f"({fallback_model}): {exc}"
            )
            self.asr_model_name = fallback_model
            model = self._load_asr_model(
                backend_name=fallback_backend,
                model_name=self.asr_model_name,
                device=self.device,
                use_fp16=self.use_fp16,
            )
            return fallback_backend, model

    @classmethod
    def _load_asr_model(cls, backend_name, model_name, device, use_fp16):
        backend_key = (backend_name, model_name, device, use_fp16)
        with cls._shared_lock:
            backend = cls._shared_asr_models.get(backend_key)
            if backend is not None:
                print(f"[Audio Init] Reusing {backend_name} {model_name} on {device}...")
                return backend

            if backend_name == "faster-whisper":
                if WhisperModel is None:
                    raise RuntimeError("faster-whisper is not installed.")
                compute_type = "float16" if use_fp16 else "float32"
                print(f"[Audio Init] Loading faster-whisper {model_name} ({compute_type})...")
                device_name = "cuda" if str(device).startswith("cuda") else str(device)
                device_index = 0
                if device_name == "cuda" and ":" in str(device):
                    try:
                        device_index = int(str(device).split(":", 1)[1])
                    except ValueError:
                        device_index = 0
                backend = WhisperModel(
                    model_name,
                    device=device_name,
                    device_index=device_index,
                    compute_type=compute_type,
                )
            elif backend_name == "whisper":
                if openai_whisper is None:
                    raise RuntimeError("openai-whisper is not installed.")
                print(f"[Audio Init] Loading Whisper {model_name}...")
                backend = openai_whisper.load_model(model_name, device=device)
            else:
                raise RuntimeError(f"Unsupported ASR backend: {backend_name}")

            cls._shared_asr_models[backend_key] = backend
            return backend

    @classmethod
    def _load_text_encoder(cls, model_name, device):
        backend_key = (model_name, device)
        with cls._shared_lock:
            backend = cls._shared_text_encoders.get(backend_key)
            if backend is None:
                print(f"[Audio Init] Loading Sentence-Transformer {model_name}...")
                backend = SentenceTransformer(model_name, device=device)
                cls._shared_text_encoders[backend_key] = backend
            else:
                print(f"[Audio Init] Reusing Sentence-Transformer {model_name} on {device}...")
        return backend

    @staticmethod
    def _get_text_embedding_dimension(text_encoder):
        dim = getattr(text_encoder, "get_sentence_embedding_dimension", lambda: None)()
        if dim:
            return int(dim)

        sample = text_encoder.encode(
            ["dimension probe"],
            convert_to_tensor=False,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        array = np.asarray(sample)
        if array.ndim == 2:
            return int(array.shape[1])
        return int(array.shape[0])

    def _build_index(self, use_fast_index):
        if use_fast_index:
            try:
                return faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            except TypeError:
                print("[Audio Warning] Current FAISS build lacks metric-aware HNSW constructor, fallback to FlatIP.")
        return faiss.IndexFlatIP(self.dimension)

    def _reset_session_state(self):
        self.index.reset()
        self.metadata = {}

    def start_new_job(self, runtime_dir):
        self.runtime_dir = runtime_dir
        os.makedirs(self.runtime_dir, exist_ok=True)
        self._reset_session_state()

    def _build_runtime_audio_path(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0] or "audio"
        cache_suffix = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.runtime_dir, f"{video_name}_{cache_suffix}.wav")

    def _extract_audio(self, video_path):
        audio_path = self._build_runtime_audio_path(video_path)
        if os.path.exists(audio_path):
            return audio_path

        print("[Audio] Extracting audio from video...")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-threads", "4",
            audio_path, "-y", "-hide_banner", "-loglevel", "error"
        ]

        try:
            subprocess.run(cmd, check=True)
            return audio_path
        except subprocess.CalledProcessError as exc:
            print(f"[Audio Warning] PCM 转码失败: {exc}")
            raise RuntimeError("ffmpeg 音频提取失败，请检查环境。")

    def _get_audio_duration(self, audio_path):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        except Exception:
            return None

    def _make_cache_path(self, video_path, language):
        try:
            stat = os.stat(video_path)
            key_src = (
                f"{video_path}|{stat.st_size}|{stat.st_mtime}|{self.asr_backend}|{self.asr_model_name}|"
                f"{self.use_vad}|{self.text_model_name}|{language}|{self.chunk_seconds}"
            )
        except FileNotFoundError:
            key_src = (
                f"{video_path}|{self.asr_backend}|{self.asr_model_name}|{self.use_vad}|"
                f"{self.text_model_name}|{language}|{self.chunk_seconds}"
            )
        cache_key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_cached_segments(self, cache_path):
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("segments", [])
        except Exception:
            return None

    def _save_cached_segments(self, cache_path, segments):
        payload = {"segments": segments}
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass

    def _split_audio(self, audio_path, chunk_seconds):
        chunk_dir = f"{audio_path}_chunks"
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_pattern = os.path.join(chunk_dir, "chunk_%03d.wav")
        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            chunk_pattern,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        subprocess.run(cmd, check=True)
        chunk_files = sorted(
            [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".wav")]
        )
        return chunk_dir, chunk_files

    def _cleanup_chunks(self, chunk_dir):
        try:
            for name in os.listdir(chunk_dir):
                path = os.path.join(chunk_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
            os.rmdir(chunk_dir)
        except Exception:
            pass

    def _build_transcribe_options(self, language):
        if self.asr_backend == "faster-whisper":
            options = {
                "beam_size": 1,
                "temperature": 0,
                "task": "transcribe",
                "condition_on_previous_text": False,
                "vad_filter": self.use_vad,
            }
            if self.use_vad:
                options["vad_parameters"] = {"min_silence_duration_ms": 500}
            if language:
                options["language"] = language
            return options

        options = {
            "beam_size": 1,
            "fp16": self.use_fp16,
            "task": "transcribe",
            "temperature": 0,
        }
        if language:
            options["language"] = language
        return options

    def _transcribe_audio_file(self, audio_path, transcribe_options):
        if self.asr_backend == "faster-whisper":
            segments, _ = self.asr_model.transcribe(audio_path, **transcribe_options)
            return [
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": str(seg.text or "").strip(),
                }
                for seg in segments
                if str(seg.text or "").strip()
            ]

        result = self.asr_model.transcribe(audio_path, **transcribe_options)
        segments = result.get("segments", [])
        return [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in segments
            if str(seg.get("text", "")).strip()
        ]

    def _transcribe_full(self, audio_path, transcribe_options):
        return self._transcribe_audio_file(audio_path, transcribe_options)

    def _transcribe_chunked(self, audio_path, transcribe_options):
        if not self.chunk_seconds:
            return self._transcribe_full(audio_path, transcribe_options)

        try:
            chunk_dir, chunk_files = self._split_audio(audio_path, self.chunk_seconds)
        except Exception as exc:
            print(f"[Audio Warning] 分段切割失败，回退为整段转录: {exc}")
            return self._transcribe_full(audio_path, transcribe_options)

        segments = []
        offset = 0.0
        try:
            for i, chunk_path in enumerate(chunk_files):
                print(f"[Audio] Transcribing chunk {i+1}/{len(chunk_files)}...")
                chunk_segments = self._transcribe_audio_file(chunk_path, transcribe_options)
                for seg in chunk_segments:
                    segments.append(
                        {
                            "start": seg["start"] + offset,
                            "end": seg["end"] + offset,
                            "text": seg["text"],
                        }
                    )
                duration = self._get_audio_duration(chunk_path)
                if duration is None:
                    duration = self.chunk_seconds
                offset += duration
        finally:
            self._cleanup_chunks(chunk_dir)

        return segments

    def process_audio(self, video_path, language=None):
        print(f"[Audio Processing] Start processing: {os.path.basename(video_path)}")
        self._reset_session_state()
        status = {
            "success": False,
            "segment_count": 0,
            "error": None,
            "warning": None,
        }

        try:
            audio_path = self._extract_audio(video_path)
        except Exception as exc:
            print(f"[Audio Error] Extraction failed: {exc}")
            status["error"] = str(exc)
            return status

        try:
            print(
                f"[Audio] Running {self.asr_backend} transcription..."
                f" model={self.asr_model_name} vad={self.use_vad}"
            )
            transcribe_options = self._build_transcribe_options(language)

            cache_path = self._make_cache_path(video_path, language)
            segments = self._load_cached_segments(cache_path)
            if segments is None:
                segments = self._transcribe_chunked(audio_path, transcribe_options)
                self._save_cached_segments(cache_path, segments)
            print(f"[Audio] Transcribed {len(segments)} segments.")

            if not segments:
                print("[Audio Warning] No speech detected.")
                status["warning"] = "未检测到可用语音，音频检索已跳过。"
                return status

            texts = [seg["text"] for seg in segments]
            print(f"[Audio] Encoding text embeddings with {self.text_model_name}...")
            embeddings = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                device=self.device,
            )
            embeddings = embeddings.cpu().numpy().astype("float32")

            self.index.add(embeddings)

            self.metadata = {}
            for i, seg in enumerate(segments):
                self.metadata[i] = {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": seg["text"].strip(),
                }

            print(f"[Audio Index] Built index with {self.index.ntotal} text segments.")
            status["success"] = True
            status["segment_count"] = self.index.ntotal
            return status
        except Exception as exc:
            print(f"[Audio Error] Processing failed: {exc}")
            status["error"] = str(exc)
            return status

    def search(self, query, k=5):
        if not query or self.index.ntotal == 0:
            return []

        print(f"[Audio Search] Query: '{query}'")
        query_vec = self.text_encoder.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )
        query_vec = query_vec.cpu().numpy().astype("float32")

        scores, indices = self.index.search(query_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.metadata:
                data = self.metadata[idx]
                score = scores[0][i]
                results.append((data["start"], data["end"], data["text"], score))

        return results

    def get_subtitles_for_timestamps(self, timestamps: list, window_seconds: float = 3.0) -> list:
        """Get subtitle/transcript text for given frame timestamps."""
        if not self.metadata:
            return [(ts, "") for ts in timestamps]

        results = []
        for ts in timestamps:
            matching_texts = []
            for data in self.metadata.values():
                seg_start = data["start"]
                seg_end = data["end"]
                if seg_start - window_seconds <= ts <= seg_end + window_seconds:
                    matching_texts.append(data["text"])

            combined_text = " ".join(matching_texts) if matching_texts else ""
            results.append((ts, combined_text))
        return results

    def get_all_transcripts(self) -> list:
        """Get all transcripts as a list of segments."""
        if not self.metadata:
            return []
        return [self.metadata[idx] for idx in sorted(self.metadata.keys())]

    def format_subtitles_for_prompt(self, timestamps: list = None, max_chars: int = 2000) -> str:
        """Format subtitles for inclusion in prompts."""
        if timestamps:
            subs = self.get_subtitles_for_timestamps(timestamps)
            texts = [text for _, text in subs if text]
        else:
            segments = self.get_all_transcripts()
            texts = [seg["text"] for seg in segments]

        combined = " ".join(texts)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        return combined
