<div align="center">

<h1>Video-RAG Ultra</h1>

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

`👁️ OpenCLIP` · `🎙️ faster-whisper` · `🧠 bge-m3` · `🤖 Qwen2.5-VL` · `🏁 Video-MME`

[快速开始](#快速开始) · [核心能力](#核心能力) · [默认配置](#当前默认配置) · [Benchmark](#benchmark) · [项目结构](#项目结构)

</div>
  
## ✨ 项目介绍

<img width="2760" height="1504" alt="video-rag-ultra-en" src="https://github.com/user-attachments/assets/a68991d9-7d5a-45fc-82c5-9267876fea98" />
<img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/3d2c348d-71e2-4267-8fab-8bf56b8a9ab4" />



Video-RAG Ultra 是一个面向长视频问答的多模态 RAG 系统，围绕“检索证据 + 多模态推理”构建。当前版本默认使用 OpenCLIP 做视觉检索、`faster-whisper large-v3` 做音频转录、`bge-m3` 做文本向量检索，并结合 `Qwen2.5-VL-7B-Instruct` 生成最终回答。

当前 Web 交互已经与代码实现保持一致：

- 🧰 左侧为任务面板：上传视频、开始分析、查看处理状态
- 💬 右侧为智能视频问答区
- 🖼️ 回答中的视觉证据卡片支持点击后在卡片内部展开关键帧图片
- 🧭 不再单独展示独立关键帧画廊

## 🚀 核心能力

| 能力 | 当前实现 |
| --- | --- |
| 👁️ 视觉检索 | 默认使用 OpenCLIP `ViT-L-14`，支持 OpenAI CLIP 回退 |
| 🎬 采样策略 | 场景切分候选 + 低帧率均匀采样，兼顾代表帧与时序连续性 |
| 🎧 音频理解 | 默认 `faster-whisper large-v3 + VAD` |
| 🧠 文本检索 | 默认 `BAAI/bge-m3` |
| 🤖 多模态问答 | 基于视觉和音频证据调用 `Qwen2.5-VL-7B-Instruct` |
| 🧾 证据可视化 | 展示时间戳、匹配分数，并支持关键帧内联预览 |
| 🏁 Benchmark | 内置 Video-MME 下载、推理与评测脚本 |

## 🧩 当前默认配置

以下内容以当前代码实现为准：

| 模块 | 默认值 |
| --- | --- |
| 视觉检索后端 | `openclip` |
| 视觉模型 | `ViT-L-14` |
| 视觉后端回退 | `clip + ViT-B/32` |
| 音频 ASR | `faster-whisper` |
| 音频模型 | `large-v3` |
| VAD | 开启 |
| 文本向量模型 | `BAAI/bge-m3` |
| VLM | `Qwen/Qwen2.5-VL-7B-Instruct` |
| MCQ 最大图片数 | `24` |
| 事件块数量 | `5~8` |

更细的默认值和环境变量见 [配置说明](#配置说明)。

## ⚡ 快速开始

### 1. 环境要求

- Python 3.10+ 推荐
- `ffmpeg`
- NVIDIA GPU 推荐，但不是强制
- 首次运行会下载模型，请保证 Hugging Face 可访问或已配置镜像/缓存

### 2. 创建环境

```bash
git clone <repository-url>
cd Video-RAG-Ultra

conda create -n video-rag-ultra python=3.10 -y
conda activate video-rag-ultra
python -m pip install -U pip setuptools wheel
```

### 3. 安装 PyTorch

如果你已经有可用的 GPU 版 PyTorch，可以跳过这一步。下面是 CUDA 12.1 的示例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

当前依赖已经包含：

- `open-clip-torch`
- `faster-whisper`
- `sentence-transformers`
- `transformers`
- `gradio`

不再需要单独安装旧版 GitHub CLIP 才能跑主流程。

### 5. 安装 ffmpeg

Ubuntu / Debian：

```bash
sudo apt-get install ffmpeg
```

macOS：

```bash
brew install ffmpeg
```

验证：

```bash
ffmpeg -version
```

### 6. 可选环境变量

```bash
# Hugging Face 缓存目录
export HF_HOME=$HOME/.cache/huggingface

# 避免 XET 下载链路问题
export HF_HUB_DISABLE_XET=1

# 国内网络可选
# export HF_ENDPOINT=https://hf-mirror.com

# 视觉检索默认配置
export VIDEO_RETRIEVER_BACKEND=openclip
export VIDEO_RETRIEVER_MODEL=ViT-L-14
export VIDEO_RETRIEVER_PRETRAINED=openai

# 音频默认配置
export AUDIO_ASR_BACKEND=faster-whisper
export AUDIO_ASR_MODEL=large-v3
export AUDIO_ASR_USE_VAD=1
export AUDIO_TEXT_EMBEDDING_MODEL=BAAI/bge-m3

# VLM
export VLM_MAX_MCQ_IMAGES=24
```

### 7. 环境验证

```bash
# CUDA 检查
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

# 视觉检索模块
python -c "from src.video_processor import VideoRetriever; r=VideoRetriever(); print(r.backend.describe(), r.dimension)"

# 音频模块（首次较慢）
python -c "from src.audio_processor import AudioRetriever; r=AudioRetriever(); print(r.asr_backend, r.asr_model_name, r.text_model_name, r.dimension)"

# VLM（首次较慢）
python -c "from src.vlm_handler import VLMHandler; v=VLMHandler(max_retries=1); print(v.available, v.device, v.load_error)"
```

### 8. 启动应用

```bash
conda activate video-rag-ultra
python src/app.py
```

可选：

```bash
HF_HOME=$HOME/.cache/huggingface python src/app.py
```

当前启动行为：

- `server_name` 固定为 `0.0.0.0`
- 未设置 `GRADIO_SERVER_PORT` 时自动选择空闲端口
- `GRADIO_SHARE` 默认关闭，不会自动生成公网链接

如果需要固定端口或打开分享：

```bash
export GRADIO_SERVER_PORT=7860
export GRADIO_SHARE=1
python src/app.py
```

## 🖥️ 使用说明

### Web 界面流程

1. 在左侧任务面板上传视频
2. 点击“开始分析”
3. 系统自动完成视觉采样、音频转录与索引构建
4. 在右侧“智能视频问答”输入问题
5. 查看回答中的证据块：
   - 视觉证据：时间戳、匹配分数、点击后展开关键帧
   - 音频证据：时间范围、相关语音文本

### 示例问题

- 视频里的人在做什么？
- 视频中有哪些关键场景？
- 这段视频的主要内容是什么？
- 这段对话里提到了哪些人物或地点？

### 前端交互

当前前端不再展示独立关键帧画廊，而是使用卡片内联交互：

- 点击回答中的某一条视觉证据卡片
- 该卡片内部原地展开对应关键帧图片
- 同时只展开一张，点击其它卡片会切换展开对象

## 🧠 检索与推理链路

### 离线索引

- `VideoRetriever.process_video()` 使用双路采样：
  - 场景切分候选
  - 低 FPS 均匀采样
- `AudioRetriever.process_audio()` 使用：
  - `faster-whisper large-v3`
  - VAD
  - `bge-m3` 文本向量编码

### 在线问答

- 文本问题触发视觉和音频检索
- 视觉结果返回时间戳、分数和关键帧路径
- 音频结果返回时间范围、文本和相似度
- `VLMHandler` 会将证据整理为时间有序 event chunks，再交给 `Qwen2.5-VL`

### Benchmark 问答

- Video-MME 默认 `question_topk_frames=24`
- `VLMHandler` 默认 `VLM_MAX_MCQ_IMAGES=24`
- 支持问题驱动选帧和长视频候选时间窗二次加密

## 🏁 Benchmark

仓库已内置 Video-MME 相关脚本。

### 常用命令

```bash
# 安装 benchmark 额外依赖
pip install datasets pandas

# 下载标注（不含视频）
python -m src.benchmark.run_videomme --download

# 下载某个 duration 的视频
python -m src.benchmark.run_videomme --download-videos short --max-workers 2

# 运行评测
python -m src.benchmark.run_videomme --durations short --with-subtitles

# 计算得分
python -m src.benchmark.eval_videomme data/videomme/results/videomme_detailed_wo_subs_*.json
```

长视频增强选帧和更多参数请查看：

- [src/benchmark/commands.md](./src/benchmark/commands.md)
- [commands.md](./commands.md)

## 🗂️ 项目结构

```text
Video-RAG-Ultra/
├── src/
│   ├── app.py
│   ├── video_processor.py
│   ├── audio_processor.py
│   ├── vlm_handler.py
│   ├── vision_backends.py
│   └── benchmark/
├── data/
│   ├── videos/
│   ├── embeddings/
│   ├── runtime/
│   └── videomme/
├── requirements.txt
├── commands.md
└── README.md
```

### 核心模块

- [src/app.py](./src/app.py)：Gradio UI、前端交互与服务编排
- [src/video_processor.py](./src/video_processor.py)：双路采样、视觉编码与检索
- [src/audio_processor.py](./src/audio_processor.py)：音频提取、转录、文本向量检索
- [src/vlm_handler.py](./src/vlm_handler.py)：Qwen2.5-VL 加载、event chunk 组织与回答生成
- [src/vision_backends.py](./src/vision_backends.py)：OpenCLIP / CLIP 后端适配层
- [src/benchmark/](./src/benchmark)：Video-MME 数据、推理和评测

## ⚙️ 配置说明

### 视觉相关

- `VIDEO_RETRIEVER_BACKEND`：默认 `openclip`
- `VIDEO_RETRIEVER_MODEL`：默认 `ViT-L-14`
- `VIDEO_RETRIEVER_PRETRAINED`：默认 `openai`
- `VIDEO_RETRIEVER_FALLBACK_MODEL`：默认回退到 `ViT-B/32`
- `VIDEO_SCENE_SAMPLE_FPS`：场景候选采样频率，默认 `2.0`

### 音频相关

- `AUDIO_ASR_BACKEND`：默认 `faster-whisper`
- `AUDIO_ASR_MODEL`：默认 `large-v3`
- `AUDIO_ASR_USE_VAD`：默认 `1`
- `AUDIO_TEXT_EMBEDDING_MODEL`：默认 `BAAI/bge-m3`
- `AUDIO_ASR_FALLBACK_MODEL`：`faster-whisper` 不可用时回退模型，默认沿用当前模型名

### VLM 相关

- `VLM_DEVICE`：默认自动优先 `cuda:1`
- `VLM_MODEL_ID`：默认 `Qwen/Qwen2.5-VL-7B-Instruct`
- `VLM_LOCAL_PATH`：默认 `./Qwen2.5-VL-7B-Instruct`
- `VLM_MAX_VISUAL_IMAGES`：默认 `6`
- `VLM_MAX_MCQ_IMAGES`：默认 `24`
- `VLM_MIN_EVENT_CHUNKS`：默认 `5`
- `VLM_MAX_EVENT_CHUNKS`：默认 `8`

### Gradio 相关

- `GRADIO_SERVER_PORT`：未设置时自动选空闲端口
- `GRADIO_SHARE`：默认关闭

## 🛠️ 常见问题

### 首次启动很慢

正常现象。第一次运行通常会发生：

- 下载 OpenCLIP / Qwen / Whisper / bge-m3
- 初始化 GPU 权重
- 建立 Hugging Face 缓存

### `AudioRetriever()` 很慢

如果使用默认 `bge-m3`，首次加载会明显慢于旧的 `all-MiniLM-L6-v2`。

如果你只想快速验证流程，可以临时回退：

```bash
AUDIO_TEXT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 python src/app.py
```

### 端口不是 7860

当前代码默认自动选空闲端口；如果需要固定端口，请设置：

```bash
export GRADIO_SERVER_PORT=7860
python src/app.py
```

### AV1 视频处理慢

当前代码会检测 AV1，并在需要时自动转码到 H.264 后再进行 OpenCV 解码，这会增加首次处理时间。

## 🙏 致谢

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Gradio](https://github.com/gradio-app/gradio)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📄 许可证

本项目采用 MIT 许可证。
