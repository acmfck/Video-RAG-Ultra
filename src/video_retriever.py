import faiss
import numpy as np
import torch
from PIL import Image

try:
    from vision_backends import build_vision_backend
except ImportError:
    from src.vision_backends import build_vision_backend


def main():
    # 1. 加载视觉后端
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载视觉后端到 {device}...")

    try:
        backend = build_vision_backend(
            backend_name="openclip",
            model_name="ViT-L-14",
            pretrained="openai",
            device=device,
        )
        print(f"视觉后端加载成功：{backend.describe()} (dim={backend.dimension})")
    except Exception as exc:
        print(f"模型加载失败: {exc}")
        return

    # 2. 准备数据 (模拟视频帧)
    print("\n正在生成模拟视频帧数据...")
    num_frames = 100
    dimension = backend.dimension

    database_vectors = np.random.random((num_frames, dimension)).astype("float32")
    faiss.normalize_L2(database_vectors)
    print(f"已生成 {num_frames} 帧的模拟特征向量。")

    # 3. 构建 FAISS 索引
    print("\n正在构建 FAISS 向量索引...")
    index = faiss.IndexFlatIP(dimension)
    index.add(database_vectors)
    print(f"索引构建完成，索引中现包含 {index.ntotal} 个向量。")

    # 4. 模拟检索
    user_query = "A red car driving on the street"
    print(f"\n用户提问: '{user_query}'")
    print("正在计算文本向量并检索...")

    image = Image.new("RGB", (224, 224), color="red")
    image_tensor = backend.preprocess_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = backend.encode_images(image_tensor)
        text_features = backend.encode_texts([user_query]).cpu().numpy().astype("float32")

    faiss.normalize_L2(text_features)
    distances, frame_indices = index.search(text_features, 5)

    print("-" * 30)
    print("检索结果 (Top-5):")
    for i, idx in enumerate(frame_indices[0]):
        print(f"Rank {i+1}: 检索到第 {idx} 帧 (相似度得分: {distances[0][i]:.4f})")
    print("-" * 30)
    print("恭喜！Video-RAG 的核心检索链路已跑通。")


if __name__ == "__main__":
    main()
