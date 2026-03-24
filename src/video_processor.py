import cv2
import clip
import torch
import faiss
import numpy as np
import os
from PIL import Image
import time
import shutil
import subprocess

class VideoRetriever:
    def __init__(self, model_name="ViT-B/32"):
        """
        Initialize retriever: load CLIP model and FAISS index
        
        Args:
            model_name: CLIP model name (default: "ViT-B/32")
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(f"[Init] 正在加载 CLIP 模型 ({model_name}) 到 {self.device}...")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            print("[Init] CLIP 模型加载成功。")
        except Exception as e:
            print(f"[Error] 模型加载失败: {e}")
            raise e
        
        self.dimension = 512 
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = {} 
        
        self.keyframe_dir = "keyframes"
        self._reset_session_state()

    def _reset_session_state(self):
        """Clear prior index entries and generated frames before processing a new video."""
        self.index.reset()
        self.metadata = {}
        if os.path.exists(self.keyframe_dir):
            try:
                shutil.rmtree(self.keyframe_dir)
            except:
                pass
        os.makedirs(self.keyframe_dir, exist_ok=True)

    def _calculate_histogram_diff(self, frame1, frame2):
        """Calculate histogram difference for keyframe detection"""
        try:
            f1_small = cv2.resize(frame1, (64, 64))
            f2_small = cv2.resize(frame2, (64, 64))

            h1 = cv2.calcHist([cv2.cvtColor(f1_small, cv2.COLOR_BGR2HSV)], [0], None, [256], [0, 256])
            h2 = cv2.calcHist([cv2.cvtColor(f2_small, cv2.COLOR_BGR2HSV)], [0], None, [256], [0, 256])
            
            cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
            
            score = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
            return score
        except Exception as e:
            return 0.0

    def _convert_to_h264(self, input_path):
        """Convert unsupported video format to H.264"""
        output_path = os.path.splitext(input_path)[0] + "_h264.mp4"
        print(f"[Auto-Fix] Converting to H.264: {os.path.basename(input_path)}")
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            output_path, "-y", "-hide_banner", "-loglevel", "error"
        ]
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(output_path):
                print("[Auto-Fix] 转码成功！")
                return output_path
        except Exception as e:
            print(f"[Error] 转码失败: {e}")
        return None

    def _embed_and_add_to_index(self, frame_buffer, timestamp_buffer, path_buffer):
        """Batch encode frames and add to FAISS index"""
        if not frame_buffer:
            return

        batch_inputs = torch.stack([self.preprocess(img) for img in frame_buffer]).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(batch_inputs)
            features = features.cpu().numpy().astype('float32')
        
        faiss.normalize_L2(features)
        
        start_id = self.index.ntotal
        self.index.add(features)
        
        for i, ts in enumerate(timestamp_buffer):
            self.metadata[start_id + i] = {
                "timestamp": ts,
                "path": path_buffer[i]
            }

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

    def _process_video_once(self, video_path, sample_rate=1, diff_threshold=0.15, max_duration_minutes=None):
        """Process a single video path without format fallback."""
        cap, fps, total_frames, duration = self._open_video(video_path)
        if cap is None:
            raise ValueError("无法读取视频，文件可能损坏。")

        print(f"[Info] Video info: FPS={fps:.2f}, Duration={duration/60:.2f} minutes")
        
        prev_valid_frame = None
        frame_buffer = []
        timestamp_buffer = []
        path_buffer = []
        batch_size = 64
        
        frame_idx = 0
        decoded_count = 0
        saved_count = 0
        step = int(fps / sample_rate) if sample_rate > 0 else 30
        
        start_time = time.time()
        
        while True:
            if step > 100: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
            else:
                if frame_idx == 0:
                    ret, frame = cap.read()
                else:
                    ret = True
                    frames_to_skip = step - 1
                    for _ in range(frames_to_skip):
                        if not cap.grab():
                            ret = False
                            break
                    if ret:
                        ret, frame = cap.retrieve()
                    else:
                        break

            if not ret:
                if decoded_count == 0:
                    print("[Warning] OpenCV 打开了视频，但无法解码任何视频帧。")
                break

            decoded_count += 1
            
            current_time_sec = frame_idx / fps
            if max_duration_minutes and (current_time_sec / 60) > max_duration_minutes:
                print(f"Reached max duration {max_duration_minutes} minutes, stopping.")
                break

            is_keyframe = False
            if prev_valid_frame is None:
                is_keyframe = True 
            else:
                diff = self._calculate_histogram_diff(prev_valid_frame, frame)
                if diff > diff_threshold:
                    is_keyframe = True
            
            if is_keyframe:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                
                frame_filename = f"frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(self.keyframe_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                frame_buffer.append(pil_image)
                timestamp_buffer.append(current_time_sec)
                path_buffer.append(frame_path)
                
                prev_valid_frame = frame
                saved_count += 1
                
                if len(frame_buffer) >= batch_size:
                    self._embed_and_add_to_index(frame_buffer, timestamp_buffer, path_buffer)
                    frame_buffer = []
                    timestamp_buffer = []
                    path_buffer = []
                    print(f"\r  -> Progress: {current_time_sec/60:.1f}/{duration/60:.1f} min (Indexed: {saved_count} frames)", end="")

            frame_idx += step

        if len(frame_buffer) > 0:
            self._embed_and_add_to_index(frame_buffer, timestamp_buffer, path_buffer)

        cap.release()
        print(f"\n[Done] Processing completed in {time.time() - start_time:.2f}s | Total indexed frames: {self.index.ntotal}")
        return decoded_count, self.index.ntotal

    def process_video(self, video_path, sample_rate=1, diff_threshold=0.15, max_duration_minutes=None):
        """
        Process video: extract keyframes, encode and index
        
        Args:
            video_path: Path to video file
            sample_rate: Frames per second to sample
            diff_threshold: Threshold for keyframe detection
            max_duration_minutes: Maximum duration to process (None for full video)
        """
        if not os.path.exists(video_path):
            parent_path = os.path.join("..", video_path)
            if os.path.exists(parent_path):
                video_path = parent_path
            else:
                raise FileNotFoundError(f"找不到视频文件: {video_path}")

        print(f"[Processing] Processing video: {os.path.basename(video_path)}")
        self._reset_session_state()

        first_error = None
        try:
            _, indexed_count = self._process_video_once(
                video_path,
                sample_rate=sample_rate,
                diff_threshold=diff_threshold,
                max_duration_minutes=max_duration_minutes,
            )
        except ValueError as e:
            first_error = e
            indexed_count = 0

        already_transcoded = os.path.splitext(video_path)[0].endswith("_h264")
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
                except ValueError as e:
                    first_error = e
                    indexed_count = 0

        if indexed_count == 0:
            if first_error is not None:
                raise first_error
            raise ValueError("No keyframes extracted!")

    def search(self, query, k=5):
        """Search for similar frames given text query"""
        print(f"\n[Search] Query: '{query}'")
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features.cpu().numpy().astype('float32')
            
        faiss.normalize_L2(text_features)
        distances, indices = self.index.search(text_features, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.metadata:
                data = self.metadata[idx]
                results.append((data["timestamp"], distances[0][i], data["path"]))
        return results

    def sample_frames_for_benchmark(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 768,
        output_dir: str = None
    ):
        """
        Extract frames at fixed FPS for benchmark evaluation (Video-MME style).
        
        Unlike process_video() which uses keyframe detection, this method
        samples frames at a constant rate for fair benchmark comparison.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to sample (default: 1.0)
            max_frames: Maximum number of frames to extract (default: 768)
            output_dir: Directory to save frames (default: self.keyframe_dir)
            
        Returns:
            List of (frame_path, timestamp) tuples
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        output_dir = output_dir or self.keyframe_dir
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        if not cap.isOpened() or video_fps <= 0:
            cap.release()
            new_path = self._convert_to_h264(video_path)
            if new_path:
                cap = cv2.VideoCapture(new_path)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / video_fps
            else:
                raise ValueError(f"Cannot open video: {video_path}")
        
        print(f"[Benchmark] Sampling {video_path}")
        print(f"[Benchmark] Duration: {duration/60:.2f}min, FPS: {fps}, Max frames: {max_frames}")
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps > 0 else 30
        
        # Calculate total frames to extract
        expected_frames = int(duration * fps)
        if expected_frames > max_frames:
            # Uniform sampling to fit within max_frames
            frame_interval = int(total_frames / max_frames)
            expected_frames = max_frames
        
        results = []
        frame_idx = 0
        saved_count = 0
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        while saved_count < max_frames:
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
            
            if frame_idx >= total_frames:
                break
        
        cap.release()
        print(f"[Benchmark] Extracted {len(results)} frames")
        
        return results

    def encode_frames_batch(self, frame_paths: list, batch_size: int = 32):
        """
        Encode a list of frame images using CLIP (for benchmark inference).
        
        Args:
            frame_paths: List of paths to frame images
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (N, 512) with L2-normalized features
        """
        all_features = []
        
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(self.preprocess(img))
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features.cpu().numpy().astype('float32')
            
            faiss.normalize_L2(features)
            all_features.append(features)
        
        return np.concatenate(all_features, axis=0) if all_features else np.array([])

    def encode_text_query(self, query: str):
        """Encode text query into normalized CLIP feature."""
        # CLIP text encoder has a strict context length (typically 77 tokens).
        # Use truncate=True to avoid runtime failures on long MCQ queries.
        text_tokens = clip.tokenize([query], truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features.cpu().numpy().astype("float32")
        faiss.normalize_L2(text_features)
        return text_features[0]

    def select_topk_frames_for_query(
        self,
        frame_paths: list,
        timestamps: list,
        frame_features,
        query: str,
        top_k: int = 24,
        neighbor_window: int = 1,
    ):
        """
        Select question-relevant frames by CLIP text-image similarity.

        Args:
            frame_paths: All sampled frame paths in chronological order
            timestamps: Frame timestamps aligned with frame_paths
            frame_features: Pre-computed frame features from encode_frames_batch()
            query: Question text (+ options) as retrieval query
            top_k: Number of top similar anchors to keep
            neighbor_window: Include +/- N neighboring frames around each anchor

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
            return frame_paths[:n], timestamps[:n]

        try:
            text_feature = self.encode_text_query(query)
        except Exception as e:
            print(f"[Warning] Failed to encode query for frame selection: {e}")
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
        selected_paths = [frame_paths[i] for i in ordered]
        selected_timestamps = [timestamps[i] for i in ordered]
        return selected_paths, selected_timestamps
