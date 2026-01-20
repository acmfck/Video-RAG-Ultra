import faiss
import numpy as np
import torch
import clip
from PIL import Image

# ==========================================
# Video-RAG 核心演示：CLIP + FAISS 语义检索
# ==========================================

def main():
    # 1. 加载 CLIP 模型
    # 自动选择设备：有显卡用 cuda，没显卡用 cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载 CLIP 模型到 {device}...")
    
    try:
        # 使用 ViT-B/32 模型，速度快，显存占用小
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP 模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 准备数据 (模拟视频帧)
    # 在真实项目中，这里应该是你用 ffmpeg 切出来的图片列表
    print("\n正在生成模拟视频帧数据...")
    num_frames = 100
    # CLIP ViT-B/32 的特征维度是 512
    dimension = 512 
    
    # 模拟 100 帧的特征向量 (注意：CLIP 使用余弦相似度，所以通常需要归一化)
    # 这里我们随机生成模拟数据
    database_vectors = np.random.random((num_frames, dimension)).astype('float32')
    faiss.normalize_L2(database_vectors) # 归一化向量，使得 L2 距离等价于余弦相似度
    
    print(f"已生成 {num_frames} 帧的模拟特征向量。")

    # 3. 构建 FAISS 索引 (Index)
    # 使用最简单的 FlatL2 索引 (暴力搜索，精度最高，适合小数据量)
    print("\n正在构建 FAISS 向量索引...")
    index = faiss.IndexFlatL2(dimension) 
    index.add(database_vectors)
    print(f"索引构建完成，索引中现包含 {index.ntotal} 个向量。")

    # 4. 模拟检索 (Retrieval)
    user_query = "A red car driving on the street"
    print(f"\n用户提问: '{user_query}'")
    print("正在计算文本向量并检索...")

    # 把文本变成向量
    # tokenize 会把文本变成 token ID
    text_tokens = clip.tokenize([user_query]).to(device)
    
    with torch.no_grad():
        # encode_text 会把 token ID 变成 512维的语义向量
        text_features = model.encode_text(text_tokens)
        text_features = text_features.cpu().numpy().astype('float32')
        
    # 归一化查询向量
    faiss.normalize_L2(text_features)

    # 搜索最相似的 Top-5 帧
    k = 5 
    # D 是距离 (Distance), I 是索引 (Index, 即第几帧)
    distances, frame_indices = index.search(text_features, k)

    print("-" * 30)
    print(f"检索结果 (Top-{k}):")
    for i, idx in enumerate(frame_indices[0]):
        # distance 越小越相似 (因为是 L2 距离，如果是内积则是越大越相似，但在归一化后是一样的趋势)
        print(f"Rank {i+1}: 检索到第 {idx} 帧 (距离得分: {distances[0][i]:.4f})")
    print("-" * 30)
    print("恭喜！Video-RAG 的核心检索链路 (CLIP + FAISS) 已跑通。")

if __name__ == "__main__":
    main()