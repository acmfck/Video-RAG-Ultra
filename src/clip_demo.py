from io import BytesIO
import os

import requests
import torch
from PIL import Image

try:
    from vision_backends import build_vision_backend
except ImportError:
    from src.vision_backends import build_vision_backend


def run_clip_demo():
    # 强制使用空闲的 GPU，例如 GPU 1。
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (Physical GPU 1)")

    print("Loading visual backend...")
    backend = build_vision_backend(
        backend_name=os.getenv("VIDEO_RETRIEVER_BACKEND", "openclip"),
        model_name=os.getenv("VIDEO_RETRIEVER_MODEL", "ViT-L-14"),
        pretrained=os.getenv("VIDEO_RETRIEVER_PRETRAINED", "openai"),
        device=device,
    )
    print(f"Loaded: {backend.describe()} (dim={backend.dimension})")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"Downloading image from {url}...")
    try:
        response = requests.get(url, timeout=15)
        image = Image.open(BytesIO(response.content))
    except Exception as exc:
        print(f"Failed to download image: {exc}")
        image = Image.new("RGB", (224, 224), color="red")
        print("Created a dummy red image.")

    image_input = backend.preprocess_image(image).unsqueeze(0).to(device)
    text_descriptions = ["a diagram", "a dog", "a cat"]

    print("Running inference...")
    with torch.no_grad():
        image_features = backend.encode_images(image_input)
        text_features = backend.encode_texts(text_descriptions)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

    print("\nResults:")
    for text, prob in zip(text_descriptions, probs[0]):
        print(f"Label: '{text}', Probability: {prob:.4f}")


if __name__ == "__main__":
    run_clip_demo()
