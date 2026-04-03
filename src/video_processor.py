import os
import shutil
import subprocess
import time
from threading import Lock

import cv2
import faiss
import numpy as np
import torch
from PIL import Image

try:
    from device_utils import resolve_device
    from vision_backends import VisionBackendError, build_vision_backend
except ImportError:
    from src.device_utils import resolve_device
    from src.vision_backends import VisionBackendError, build_vision_backend


class VideoRetriever:
    _shared_lock = Lock()
    _shared_backends = {}

    def __init__(
        self,
        model_name=None,
        keyframe_dir="keyframes",
        backend_name=None,
        pretrained=None,
    ):
        """
        Initialize retriever with a configurable vision backend and FAISS index.

        Args:
            model_name: Backend model name. Defaults to OpenCLIP ViT-L-14.
            keyframe_dir: Runtime directory for extracted frames.
            backend_name: Vision backend name, e.g. "openclip" or "clip".
            pretrained: Optional OpenCLIP pretrained tag.
        """
        self.device = resolve_device(preferred_cuda_indices=[0], fallback="cpu")
        self.backend = self._load_backend(
            backend_name=backend_name,
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.backend_name = self.backend.backend_name
        self.model_name = self.backend.model_name
        self.pretrained = self.backend.pretrained
        self.model = self.backend.model
        self.preprocess = self.backend.preprocess_image

        self.dimension = int(self.backend.dimension)
        self.index = self._build_index()
        self.metadata = {}

        self.keyframe_dir = keyframe_dir
        self._reset_session_state()

    @classmethod
    def _load_backend(cls, backend_name, model_name, pretrained, device):
        requested_backend = (
            backend_name
            or os.getenv("VIDEO_RETRIEVER_BACKEND")
            or ("clip" if model_name and "/" in model_name else "openclip")
        )
        requested_model = model_name or os.getenv(
            "VIDEO_RETRIEVER_MODEL",
            "ViT-L-14" if requested_backend == "openclip" else "ViT-B/32",
        )
        requested_pretrained = pretrained or os.getenv("VIDEO_RETRIEVER_PRETRAINED", "openai")
        cache_key = (requested_backend, requested_model, requested_pretrained, device)

        with cls._shared_lock:
            backend = cls._shared_backends.get(cache_key)
            if backend is not None:
                print(f"[Init] 复用视觉后端 {backend.describe()} on {device}。")
                return backend

            print(f"[Init] 正在加载视觉后端 {requested_backend}:{requested_model} 到 {device}...")
            try:
                backend = build_vision_backend(
                    backend_name=requested_backend,
                    model_name=requested_model,
                    pretrained=requested_pretrained,
                    device=device,
                )
            except VisionBackendError as exc:
                if requested_backend == "clip":
                    print(f"[Error] 视觉后端加载失败: {exc}")
                    raise

                fallback_model = os.getenv("VIDEO_RETRIEVER_FALLBACK_MODEL", "ViT-B/32")
                print(
                    f"[Warning] OpenCLIP 后端不可用，将回退到 OpenAI CLIP ({fallback_model})。"
                )
                backend = build_vision_backend(
                    backend_name="clip",
                    model_name=fallback_model,
                    pretrained=None,
                    device=device,
                )

            cls._shared_backends[cache_key] = backend
            print(
                f"[Init] 视觉后端加载成功：{backend.describe()} | "
                f"feature_dim={backend.dimension}"
            )
            return backend

    def _build_index(self):
        return faiss.IndexFlatIP(self.dimension)

    def start_new_job(self, keyframe_dir):
        """Switch runtime artifacts to a new job-specific directory."""
        self.keyframe_dir = keyframe_dir
        self._reset_session_state()

    def _reset_session_state(self):
        """Clear prior index entries and generated frames before processing a new video."""
        self.index.reset()
        self.metadata = {}
        if os.path.exists(self.keyframe_dir):
            try:
                shutil.rmtree(self.keyframe_dir)
            except OSError:
                pass
        os.makedirs(self.keyframe_dir, exist_ok=True)

    def _calculate_histogram_diff(self, frame1, frame2):
        """Calculate histogram difference for lightweight scene-cut detection."""
        try:
            f1_small = cv2.resize(frame1, (64, 64))
            f2_small = cv2.resize(frame2, (64, 64))

            h1 = cv2.calcHist([cv2.cvtColor(f1_small, cv2.COLOR_BGR2HSV)], [0], None, [256], [0, 256])
            h2 = cv2.calcHist([cv2.cvtColor(f2_small, cv2.COLOR_BGR2HSV)], [0], None, [256], [0, 256])

            cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)

            return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        except Exception:
            return 0.0

    def _convert_to_h264(self, input_path):
        """Convert unsupported video format to H.264."""
        output_path = os.path.splitext(input_path)[0] + "_h264.mp4"
        print(f"[Auto-Fix] Converting to H.264: {os.path.basename(input_path)}")
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            output_path, "-y", "-hide_banner", "-loglevel", "error"
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if os.path.exists(output_path):
                print("[Auto-Fix] 转码成功！")
                return output_path
            stderr_text = (result.stderr or "").strip()
            if stderr_text:
                print("[Auto-Fix] 转码完成，但 ffmpeg 返回了额外日志。")
        except subprocess.CalledProcessError as exc:
            print("[Error] 转码失败。")
            stderr_text = (exc.stderr or "").strip()
            if stderr_text:
                print(stderr_text)
        except Exception as exc:
            print(f"[Error] 转码失败: {exc}")
        return None

    def _probe_video_codec(self, video_path):
        """Best-effort probe of the first video stream codec."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip().lower() or None
        except Exception:
            return None

    def _open_video(self, video_path):
        """Open a video and return metadata for downstream frame extraction."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            cap.release()
            return None, 0.0, 0, 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        return cap, fps, total_frames, duration

    def _open_video_with_fallback(self, video_path):
        """Open a video, converting to H.264 if needed."""
        cap, fps, total_frames, duration = self._open_video(video_path)
        if cap is not None:
            return cap, fps, total_frames, duration, video_path

        new_path = self._convert_to_h264(video_path)
        if not new_path:
            raise ValueError(f"Cannot open video: {video_path}")

        cap, fps, total_frames, duration = self._open_video(new_path)
        if cap is None:
            raise ValueError(f"Cannot open transcoded video: {new_path}")
        return cap, fps, total_frames, duration, new_path

    def _embed_and_add_to_index(self, frame_buffer, timestamp_buffer, path_buffer, source_buffer=None):
        """Batch encode frames and add them to the FAISS index."""
        if not frame_buffer:
            return

        batch_inputs = torch.stack([self.preprocess(img) for img in frame_buffer]).to(self.device)

        features = self.backend.encode_images(batch_inputs)
        features = features.detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(features)

        start_id = self.index.ntotal
        self.index.add(features)

        for i, ts in enumerate(timestamp_buffer):
            metadata = {
                "timestamp": ts,
                "path": path_buffer[i],
            }
            if source_buffer:
                metadata["source"] = source_buffer[i]
            self.metadata[start_id + i] = metadata

    def _encode_text_features(self, queries):
        text_features = self.backend.encode_texts(list(queries))
        text_features = text_features.detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(text_features)
        return text_features

    @staticmethod
    def _select_evenly_spaced_indices(total_items, keep_items):
        if total_items <= 0:
            return []
        if keep_items is None or keep_items <= 0 or keep_items >= total_items:
            return list(range(total_items))
        if keep_items == 1:
            return [0]

        last_idx = total_items - 1
        indices = []
        for i in range(keep_items):
            idx = round(i * last_idx / (keep_items - 1))
            if not indices or idx > indices[-1]:
                indices.append(idx)

        idx = total_items - 1
        while len(indices) < keep_items and idx >= 0:
            if idx not in indices:
                indices.append(idx)
            idx -= 1

        return sorted(indices[:keep_items])

    def _read_frame_at_index(self, cap, target_idx, current_idx, step_hint):
        if target_idx < 0:
            return False, None

        if target_idx == 0 or current_idx is None or target_idx <= current_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            return cap.read()

        frames_to_skip = target_idx - current_idx - 1
        if step_hint > 100 or frames_to_skip > 100:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            return cap.read()

        ret = True
        for _ in range(frames_to_skip):
            if not cap.grab():
                ret = False
                break
        if not ret:
            return False, None
        return cap.retrieve()

    def _persist_candidate_frame(self, frame, timestamp, saved_count, source):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        frame_filename = f"frame_{saved_count:05d}.jpg"
        frame_path = os.path.join(self.keyframe_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        return pil_image, frame_path

    def _process_video_once(self, video_path, sample_rate=1, diff_threshold=0.15, max_duration_minutes=None):
        """Process a single video path without format fallback."""
        cap, fps, total_frames, duration = self._open_video(video_path)
        if cap is None:
            raise ValueError("无法读取视频，文件可能损坏。")

        print(f"[Info] Video info: FPS={fps:.2f}, Duration={duration/60:.2f} minutes")

        uniform_sample_fps = float(max(sample_rate or 0.0, 0.1))
        scene_sample_fps = max(
            uniform_sample_fps,
            float(os.getenv("VIDEO_SCENE_SAMPLE_FPS", "2.0")),
        )
        min_gap_seconds = float(os.getenv("VIDEO_MIN_CANDIDATE_GAP_SECONDS", "0.35"))

        uniform_step = max(1, int(round(fps / uniform_sample_fps)))
        scene_step = max(1, int(round(fps / scene_sample_fps)))
        base_step = max(1, min(uniform_step, scene_step))

        frame_buffer = []
        timestamp_buffer = []
        path_buffer = []
        source_buffer = []
        batch_size = 64

        frame_idx = 0
        decoded_count = 0
        saved_count = 0
        current_idx = None
        next_uniform_frame = 0
        prev_scene_frame = None
        last_saved_ts = None

        start_time = time.time()

        while frame_idx < total_frames:
            ret, frame = self._read_frame_at_index(cap, frame_idx, current_idx, base_step)
            if not ret:
                if decoded_count == 0:
                    print("[Warning] OpenCV 打开了视频，但无法解码任何视频帧。")
                break

            decoded_count += 1
            current_idx = frame_idx
            current_time_sec = frame_idx / fps
            if max_duration_minutes and (current_time_sec / 60) > max_duration_minutes:
                print(f"Reached max duration {max_duration_minutes} minutes, stopping.")
                break

            is_uniform = frame_idx >= next_uniform_frame
            if is_uniform:
                while next_uniform_frame <= frame_idx:
                    next_uniform_frame += uniform_step

            is_scene = prev_scene_frame is None
            if prev_scene_frame is not None:
                diff = self._calculate_histogram_diff(prev_scene_frame, frame)
                if diff > diff_threshold:
                    is_scene = True
            prev_scene_frame = frame

            if is_uniform or is_scene:
                if last_saved_ts is None or abs(current_time_sec - last_saved_ts) >= min_gap_seconds:
                    source = "scene+uniform" if is_uniform and is_scene else ("scene" if is_scene else "uniform")
                    pil_image, frame_path = self._persist_candidate_frame(frame, current_time_sec, saved_count, source)
                    frame_buffer.append(pil_image)
                    timestamp_buffer.append(current_time_sec)
                    path_buffer.append(frame_path)
                    source_buffer.append(source)
                    last_saved_ts = current_time_sec
                    saved_count += 1

                    if len(frame_buffer) >= batch_size:
                        self._embed_and_add_to_index(frame_buffer, timestamp_buffer, path_buffer, source_buffer)
                        frame_buffer = []
                        timestamp_buffer = []
                        path_buffer = []
                        source_buffer = []
                        print(
                            f"\r  -> Progress: {current_time_sec/60:.1f}/{duration/60:.1f} min "
                            f"(Indexed: {saved_count} frames)",
                            end="",
                        )

            frame_idx += base_step

        if frame_buffer:
            self._embed_and_add_to_index(frame_buffer, timestamp_buffer, path_buffer, source_buffer)

        cap.release()
        print(
            f"\n[Done] Processing completed in {time.time() - start_time:.2f}s | "
            f"Total indexed frames: {self.index.ntotal}"
        )
        return decoded_count, self.index.ntotal

    def process_video(self, video_path, sample_rate=1, diff_threshold=0.15, max_duration_minutes=None):
        """
        Process video: extract dual-route candidates, encode and index.

        Args:
            video_path: Path to video file
            sample_rate: Low-FPS uniform sampling rate
            diff_threshold: Threshold for lightweight scene-cut detection
            max_duration_minutes: Maximum duration to process (None for full video)
        """
        if not os.path.exists(video_path):
            parent_path = os.path.join("..", video_path)
            if os.path.exists(parent_path):
                video_path = parent_path
            else:
                raise FileNotFoundError(f"找不到视频文件: {video_path}")

        print(f"[Processing] Processing video: {os.path.basename(video_path)}")
        print(
            f"[Processing] Dual-route sampling enabled | "
            f"uniform_fps={sample_rate} "
            f"scene_fps={os.getenv('VIDEO_SCENE_SAMPLE_FPS', '2.0')}"
        )
        self._reset_session_state()

        already_transcoded = os.path.splitext(video_path)[0].endswith("_h264")
        codec_name = self._probe_video_codec(video_path)
        if codec_name == "av1" and not already_transcoded:
            print("[Info] Detected AV1 video stream, transcoding to H.264 before OpenCV decode...")
            new_path = self._convert_to_h264(video_path)
            if new_path:
                video_path = new_path
                already_transcoded = True
                self._reset_session_state()

        first_error = None
        try:
            _, indexed_count = self._process_video_once(
                video_path,
                sample_rate=sample_rate,
                diff_threshold=diff_threshold,
                max_duration_minutes=max_duration_minutes,
            )
        except ValueError as exc:
            first_error = exc
            indexed_count = 0

        if indexed_count == 0 and not already_transcoded:
            retry_reason = "OpenCV 读取失败" if first_error else "OpenCV 无法解码实际视频帧"
            print(f"[Warning] {retry_reason}，尝试自动转码...")
            new_path = self._convert_to_h264(video_path)
            if new_path:
                self._reset_session_state()
                print(f"[Retry] Reprocessing transcoded video: {os.path.basename(new_path)}")
                try:
                    _, indexed_count = self._process_video_once(
                        new_path,
                        sample_rate=sample_rate,
                        diff_threshold=diff_threshold,
                        max_duration_minutes=max_duration_minutes,
                    )
                except ValueError as exc:
                    first_error = exc
                    indexed_count = 0

        if indexed_count == 0:
            if first_error is not None:
                raise first_error
            raise ValueError("No visual candidates extracted!")

    def search(self, query, k=5):
        """Search for similar frames given a text query."""
        if not query or self.index.ntotal == 0:
            return []

        print(f"\n[Search] Query: '{query}'")
        text_features = self._encode_text_features([query])
        scores, indices = self.index.search(text_features, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.metadata:
                data = self.metadata[idx]
                results.append((data["timestamp"], scores[0][i], data["path"]))
        return results

    def sample_frames_for_benchmark(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 768,
        output_dir: str = None,
    ):
        """
        Extract frames at fixed FPS for benchmark evaluation (Video-MME style).

        Unlike process_video(), this method keeps fixed-rate sampling for benchmark fairness.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = output_dir or self.keyframe_dir
        os.makedirs(output_dir, exist_ok=True)

        cap, video_fps, total_frames, duration, resolved_path = self._open_video_with_fallback(video_path)

        print(f"[Benchmark] Sampling {resolved_path}")
        print(f"[Benchmark] Duration: {duration/60:.2f}min, FPS: {fps}, Max frames: {max_frames}")

        frame_interval = max(1, int(video_fps / fps)) if fps > 0 else max(1, int(video_fps))
        expected_frames = int(duration * fps)
        if expected_frames > max_frames and max_frames > 0:
            frame_interval = max(1, int(total_frames / max_frames))

        results = []
        frame_idx = 0
        saved_count = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        while saved_count < max_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / video_fps
            frame_filename = f"{video_name}_frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            cv2.imwrite(frame_path, frame)
            results.append((frame_path, timestamp))

            saved_count += 1
            frame_idx += frame_interval

        cap.release()
        print(f"[Benchmark] Extracted {len(results)} frames")
        return results

    def encode_frames_batch(self, frame_paths: list, batch_size: int = 32):
        """Encode a list of frame images using the active visual backend."""
        all_features = []

        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(self.preprocess(img))

            batch_tensor = torch.stack(batch_images).to(self.device)
            features = self.backend.encode_images(batch_tensor)
            features = features.detach().cpu().numpy().astype("float32")

            faiss.normalize_L2(features)
            all_features.append(features)

        if not all_features:
            return np.array([], dtype="float32")
        return np.concatenate(all_features, axis=0)

    def encode_text_query(self, query: str):
        """Encode a text query into a normalized backend feature."""
        return self._encode_text_features([query])[0]

    @staticmethod
    def _merge_windows(windows):
        if not windows:
            return []
        windows = sorted((max(0.0, start), max(start, end)) for start, end in windows)
        merged = [windows[0]]
        for start, end in windows[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + 0.25:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _sample_frames_in_windows(
        self,
        video_path,
        windows,
        sample_fps,
        max_frames,
        output_dir,
        prefix="dense",
    ):
        if not windows or sample_fps <= 0 or max_frames <= 0:
            return []

        cap, video_fps, total_frames, _, resolved_path = self._open_video_with_fallback(video_path)
        output_dir = output_dir or self.keyframe_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamps = []
        step = 1.0 / sample_fps
        for start, end in windows:
            current = start
            while current <= end + 1e-6:
                timestamps.append(round(current, 3))
                current += step

        timestamps = sorted(dict.fromkeys(timestamps))
        if len(timestamps) > max_frames:
            keep = self._select_evenly_spaced_indices(len(timestamps), max_frames)
            timestamps = [timestamps[idx] for idx in keep]

        frames = []
        for idx, timestamp in enumerate(timestamps):
            frame_idx = min(total_frames - 1, max(0, int(round(timestamp * video_fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_filename = f"{prefix}_{idx:03d}_{int(timestamp * 1000):08d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frames.append((frame_path, frame_idx / video_fps))

        cap.release()
        if frames:
            print(
                f"[Benchmark] Dense sampled {len(frames)} frames from {len(windows)} candidate windows "
                f"@ {sample_fps}fps ({resolved_path})"
            )
        return frames

    def select_topk_frames_for_query(
        self,
        frame_paths: list,
        timestamps: list,
        frame_features,
        query: str,
        top_k: int = 24,
        neighbor_window: int = 1,
        video_path: str = None,
        dense_sample_fps: float = 0.0,
        dense_window_seconds: float = 0.0,
        dense_max_frames: int = 0,
        dense_output_dir: str = None,
    ):
        """
        Select question-relevant frames by text-image similarity and optional dense resampling.

        Returns:
            Tuple(selected_frame_paths, selected_timestamps)
        """
        n = min(len(frame_paths), len(timestamps))
        if n == 0:
            return [], []

        if frame_features is None or len(frame_features) == 0:
            return frame_paths[:n], timestamps[:n]

        features = np.asarray(frame_features)
        if features.ndim != 2 or features.shape[0] < n:
            return frame_paths[:n], timestamps[:n]
        features = features[:n]

        if not query or top_k <= 0 or top_k >= n:
            base_paths = frame_paths[:n]
            base_timestamps = timestamps[:n]
        else:
            try:
                text_feature = self.encode_text_query(query)
            except Exception as exc:
                print(f"[Warning] Failed to encode query for frame selection: {exc}")
                return frame_paths[:n], timestamps[:n]

            scores = features @ text_feature
            top_k = min(top_k, n)
            anchor_indices = np.argsort(-scores)[:top_k]

            selected = set(int(i) for i in anchor_indices)
            neighbor_window = max(0, int(neighbor_window))
            if neighbor_window > 0:
                for idx in list(selected):
                    left = max(0, idx - neighbor_window)
                    right = min(n, idx + neighbor_window + 1)
                    for j in range(left, right):
                        selected.add(j)

            ordered = sorted(selected)
            base_paths = [frame_paths[i] for i in ordered]
            base_timestamps = [timestamps[i] for i in ordered]

            if (
                video_path
                and dense_sample_fps > 0
                and dense_window_seconds > 0
                and dense_max_frames > 0
                and len(anchor_indices) > 0
            ):
                anchor_indices = list(anchor_indices[: min(4, len(anchor_indices))])
                half_window = dense_window_seconds / 2.0
                dense_windows = self._merge_windows([
                    (timestamps[idx] - half_window, timestamps[idx] + half_window)
                    for idx in anchor_indices
                ])
                dense_frames = self._sample_frames_in_windows(
                    video_path=video_path,
                    windows=dense_windows,
                    sample_fps=dense_sample_fps,
                    max_frames=dense_max_frames,
                    output_dir=dense_output_dir,
                    prefix="dense_query",
                )
                if dense_frames:
                    combined = list(zip(base_paths, base_timestamps))
                    combined.extend(dense_frames)
                    dedup = {}
                    for path, ts in combined:
                        key = (round(float(ts), 3), path)
                        dedup[key] = (path, ts)
                    ordered_pairs = sorted(dedup.values(), key=lambda x: (x[1], x[0]))
                    base_paths = [path for path, _ in ordered_pairs]
                    base_timestamps = [ts for _, ts in ordered_pairs]

        return base_paths, base_timestamps
