import whisper
import faiss
import numpy as np
import os
import torch
import subprocess
from sentence_transformers import SentenceTransformer

class AudioRetriever:
    def __init__(self, whisper_model_size="medium", use_fp16=True, use_fast_index=False):
        """
        Args:
            whisper_model_size: "tiny", "base", "small", "medium", "large-v3"
                - medium: 平衡速度和精度（推荐）
                - small: 更快，精度略降
                - large-v3: 最准确但最慢
            use_fp16: 使用半精度加速
            use_fast_index: 使用 HNSW 索引（大数据量时更快）
        """
        self.device = "cuda:2" if torch.cuda.device_count() > 2 else "cuda:0"
        print(f"[Audio Init] Loading models on {self.device}...")
        
        # 1. 加载 Whisper（可选择更小的模型）
        print(f"[Audio Init] Loading Whisper {whisper_model_size}...")
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
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
        
        result = self.whisper_model.transcribe(audio_path, **transcribe_options)
        segments = result["segments"]
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