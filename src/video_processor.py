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
        if os.path.exists(self.keyframe_dir):
            try:
                shutil.rmtree(self.keyframe_dir)
            except:
                pass
        if not os.path.exists(self.keyframe_dir):
            os.makedirs(self.keyframe_dir)

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
            "-c:v", "libx264", "-c:a", "copy",
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
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not cap.isOpened() or fps <= 0:
            print(f"[Warning] OpenCV 读取失败，尝试自动转码...")
            cap.release()
            new_path = self._convert_to_h264(video_path)
            if new_path:
                video_path = new_path
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or fps <= 0:
                raise ValueError("无法读取视频，文件可能损坏。")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"[Info] Video info: FPS={fps:.2f}, Duration={duration/60:.2f} minutes")
        
        prev_valid_frame = None
        frame_buffer = []
        timestamp_buffer = []
        path_buffer = []
        batch_size = 64
        
        frame_idx = 0
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

            if not ret: break
            
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
        
        if self.index.ntotal == 0:
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