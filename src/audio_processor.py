import whisper
import faiss
import numpy as np
import os
import json
import hashlib
import torch
import subprocess
from sentence_transformers import SentenceTransformer

class AudioRetriever:
    def __init__(
        self,
        whisper_model_size="medium",
        use_fp16=True,
        use_fast_index=False,
        chunk_seconds=300,
        cache_dir="../data/embeddings/audio_cache",
    ):
        """
        Args:
            whisper_model_size: "tiny", "base", "small", "medium", "large-v3"
                - medium: 平衡速度和精度（推荐）
                - small: 更快，精度略降
                - large-v3: 最准确但最慢
            use_fp16: 使用半精度加速
            use_fast_index: 使用 HNSW 索引（大数据量时更快）
        """
        # GPU分配策略：
        # - 3+ GPU: 使用独立的GPU 2
        # - 2 GPU: 使用GPU 0（与CLIP共享，Qwen-VL独占GPU 1）
        # - 1 GPU: 使用GPU 0
        if torch.cuda.device_count() >= 3:
            self.device = "cuda:2"  # 3个或更多GPU时，使用独立的GPU 2
        elif torch.cuda.device_count() >= 2:
            self.device = "cuda:0"  # 2个GPU时，与CLIP共享GPU 0（Qwen-VL独占GPU 1）
        else:
            self.device = "cuda:0"  # 只有1个GPU时，使用GPU 0
        print(f"[Audio Init] Loading models on {self.device} (Total GPUs: {torch.cuda.device_count()})...")
        self.whisper_model_size = whisper_model_size
        
        # 1. 加载 Whisper（可选择更小的模型）
        print(f"[Audio Init] Loading Whisper {whisper_model_size}...")
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.chunk_seconds = chunk_seconds
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 2. 加载文本向量模型
        print("[Audio Init] Loading Sentence-Transformer...")
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # 3. 初始化 FAISS
        self.dimension = 384
        if use_fast_index:
            # HNSW 索引，检索更快（适合 >1000 条数据）
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # 简单索引，构建快（适合 <1000 条数据）
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.metadata = {}
    
    def _extract_audio(self, video_path):
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        if os.path.exists(audio_path):
            return audio_path
            
        print(f"[Audio] Extracting audio from video...")
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
        except subprocess.CalledProcessError as e:
            print(f"[Audio Warning] PCM 转码失败: {e}")
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
            key_src = f"{video_path}|{stat.st_size}|{stat.st_mtime}|{self.whisper_model_size}|{language}|{self.chunk_seconds}"
        except FileNotFoundError:
            key_src = f"{video_path}|{self.whisper_model_size}|{language}|{self.chunk_seconds}"
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
        payload = {
            "segments": segments,
        }
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

    def _transcribe_full(self, audio_path, transcribe_options):
        result = self.whisper_model.transcribe(audio_path, **transcribe_options)
        segments = result.get("segments", [])
        return [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in segments
        ]

    def _transcribe_chunked(self, audio_path, transcribe_options):
        if not self.chunk_seconds:
            return self._transcribe_full(audio_path, transcribe_options)

        try:
            chunk_dir, chunk_files = self._split_audio(audio_path, self.chunk_seconds)
        except Exception as e:
            print(f"[Audio Warning] 分段切割失败，回退为整段转录: {e}")
            return self._transcribe_full(audio_path, transcribe_options)

        segments = []
        offset = 0.0
        try:
            for i, chunk_path in enumerate(chunk_files):
                print(f"[Audio] Transcribing chunk {i+1}/{len(chunk_files)}...")
                result = self.whisper_model.transcribe(chunk_path, **transcribe_options)
                for seg in result.get("segments", []):
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
        
        # 1. 提取音频
        try:
            audio_path = self._extract_audio(video_path)
        except Exception as e:
            print(f"[Audio Error] Extraction failed: {e}")
            return
        
        # 2. Whisper 转录（优化参数）
        print("[Audio] Running Whisper transcription...")
        transcribe_options = {
            "beam_size": 1,  # 从 5 降到 1，速度提升 3-5x
            "fp16": self.use_fp16,  # 半精度加速
            "task": "transcribe",
            "temperature": 0,
        }
        if language:
            transcribe_options["language"] = language

        cache_path = self._make_cache_path(video_path, language)
        segments = self._load_cached_segments(cache_path)
        if segments is None:
            segments = self._transcribe_chunked(audio_path, transcribe_options)
            self._save_cached_segments(cache_path, segments)
        print(f"[Audio] Transcribed {len(segments)} segments.")
        
        if not segments:
            print("[Audio Warning] No speech detected.")
            return
        
        # 3. 批量编码文本向量（优化）
        texts = [seg["text"] for seg in segments]
        print("[Audio] Encoding text embeddings...")
        embeddings = self.text_encoder.encode(
            texts,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # 自动归一化
            device=self.device
        )
        embeddings = embeddings.cpu().numpy().astype('float32')
        
        # 4. 存入索引
        self.index.reset()
        self.index.add(embeddings)
        
        # 5. 保存元数据
        self.metadata = {}
        for i, seg in enumerate(segments):
            self.metadata[i] = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
        
        print(f"[Audio Index] Built index with {self.index.ntotal} text segments.")
    
    def search(self, query, k=5):
        print(f"[Audio Search] Query: '{query}'")
        
        query_vec = self.text_encoder.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device
        )
        query_vec = query_vec.cpu().numpy().astype('float32')
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.metadata:
                data = self.metadata[idx]
                score = distances[0][i]
                results.append((data['start'], data['text'], score))
        
        return results
