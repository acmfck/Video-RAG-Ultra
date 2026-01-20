import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import os

def run_clip_demo():
    # 强制使用空闲的 GPU，例如 GPU 1
    # 注意：设置 CUDA_VISIBLE_DEVICES 后，代码内部看到的 "cuda:0" 实际上对应的就是物理 GPU 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # 检查是否有 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (Physical GPU 1)")

    # 加载模型
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 准备示例图片 (从网络加载一张猫的图片)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"Downloading image from {url}...")
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Failed to download image: {e}")
        # 如果下载失败，创建一个纯色图片作为 fallback
        image = Image.new('RGB', (224, 224), color='red')
        print("Created a dummy red image.")

    # 预处理图片
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 准备文本描述
    text_descriptions = ["a diagram", "a dog", "a cat"]
    text_tokens = clip.tokenize(text_descriptions).to(device)

    # 推理
    print("Running inference...")
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        
        # 计算 logits 和 probabilities
        logits_per_image, logits_per_text = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("\nResults:")
    for text, prob in zip(text_descriptions, probs[0]):
        print(f"Label: '{text}', Probability: {prob:.4f}")

if __name__ == "__main__":
    run_clip_demo()
