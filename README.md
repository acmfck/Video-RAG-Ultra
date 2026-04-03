# Video-RAG Ultra

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

多模态视频理解与问答助手

上传视频后自动构建视觉与音频索引，基于证据完成视频问答。

[快速开始](#快速开始) · [当前默认配置](#当前默认配置) · [使用说明](#使用说明) · [Benchmark](#benchmark) · [项目结构](#项目结构)

</div>

---

## 项目介绍

![video rag ultra](./video%20rag%20ultra.jpg)

Video-RAG Ultra 是一个面向长视频问答的多模态 RAG 系统。当前版本通过：

- OpenCLIP 视觉检索
- faster-whisper 音频转录
- bge-m3 文本向量检索
- Qwen2.5-VL 视觉语言模型

来完成视频内容理解、证据检索与回答生成。

当前前端交互已经对齐到现有代码：

- 左侧是任务面板：上传视频、开始分析、查看处理状态
- 右侧是智能视频问答
- 回答中的“视觉证据”卡片可点击，并在卡片内部直接展开对应关键帧图片
- 不再使用单独的关键帧画廊区域

---

## 核心能力

- 视觉检索：默认使用 OpenCLIP `ViT-L-14`，并保留 OpenAI CLIP 回退路径
- 双路采样：场景切分候选 + 低帧率均匀采样，兼顾代表帧与时序连续性
- 音频理解：默认 `faster-whisper large-v3 + VAD`
- 文本检索：默认 `BAAI/bge-m3`
- 多模态问答：基于检索到的视觉和音频证据调用 `Qwen2.5-VL-7B-Instruct`
- 证据可视化：回答中显示视觉/音频证据时间戳、匹配分数，并支持关键帧内联预览
- Benchmark 支持：内置 Video-MME 下载、推理与评测脚本

---

## 当前默认配置

以下内容来自当前代码实现，而不是历史文档：

| 模块 | 当前默认值 | 代码位置 |
| --- | --- | --- |
| 视觉检索后端 | `openclip` | [src/video_processor.py](./src/video_processor.py) |
| 视觉模型 | `ViT-L-14` | [src/video_processor.py](./src/video_processor.py) |
| 视觉后端回退 | `clip + ViT-B/32` | [src/video_processor.py](./src/video_processor.py) |
| 音频 ASR | `faster-whisper` | [src/audio_processor.py](./src/audio_processor.py) |
| 音频模型 | `large-v3` | [src/audio_processor.py](./src/audio_processor.py) |
| VAD | 开启 | [src/audio_processor.py](./src/audio_processor.py) |
| 文本向量模型 | `BAAI/bge-m3` | [src/audio_processor.py](./src/audio_processor.py) |
| VLM | `Qwen/Qwen2.5-VL-7B-Instruct` | [src/vlm_handler.py](./src/vlm_handler.py) |
| MCQ 最大图片数 | `24` | [src/vlm_handler.py](./src/vlm_handler.py) |
| 事件块数量 | `5~8` | [src/vlm_handler.py](./src/vlm_handler.py) |

---

## 快速开始

### 环境要求

- Python 3.10+ 推荐
- `ffmpeg`
- NVIDIA GPU 推荐，但不是强制
- 首次运行会下载模型，需保证 Hugging Face 可访问或已配置镜像/缓存

### 1. 创建环境

```bash
git clone <repository-url>
cd Video-RAG-Ultra

conda create -n video-rag-ultra python=3.10 -y
conda activate video-rag-ultra
python -m pip install -U pip setuptools wheel
```

### 2. 安装 PyTorch

如果你已经有合适的 GPU 版 PyTorch，可以跳过这一步。

示例：CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果你使用 CPU 或其它 CUDA 版本，请按 PyTorch 官方方式安装对应版本。

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

当前依赖已经包含：

- `open-clip-torch`
- `faster-whisper`
- `sentence-transformers`
- `transformers`
- `gradio`

不再需要单独从 GitHub 安装旧版 OpenAI CLIP 才能跑主流程。

### 4. 可选环境变量

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

### 5. 环境验证

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

### 6. 启动应用

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
- 如果没有设置 `GRADIO_SERVER_PORT`，Gradio 会自动选择空闲端口
- `GRADIO_SHARE` 默认关闭，不会自动生成公网链接

如果需要固定端口或打开分享：

```bash
export GRADIO_SERVER_PORT=7860
export GRADIO_SHARE=1
python src/app.py
```

---

## 使用说明

### Web 界面流程

1. 在左侧“任务面板”上传视频
2. 点击“开始分析”
3. 系统会自动：
   - 执行双路视觉采样并建立视觉索引
   - 抽取音频并进行转录
   - 建立文本向量索引
4. 在右侧“智能视频问答”输入问题
5. 查看回答中的证据块：
   - 视觉证据：时间戳、匹配分数、点击后展开关键帧
   - 音频证据：时间范围、相关语音文本

### 示例问题

- 视频里的人在做什么？
- 视频中有哪些关键场景？
- 这段视频的主要内容是什么？
- 这段对话里提到了哪些人物或地点？

### 前端说明

当前前端不再显示独立关键帧画廊。

现在的交互方式是：

- 在回答中的“视觉证据”列表里点击某一条证据卡片
- 该卡片内部原地展开对应关键帧图片
- 同时只展开一张，点击其它条目会切换展开对象

---

## 检索与推理链路

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
- `VLMHandler` 将证据整理为时间有序 event chunks 后交给 `Qwen2.5-VL`

### Benchmark 问答

- Video-MME 默认 `question_topk_frames=24`
- `VLMHandler` 默认 `VLM_MAX_MCQ_IMAGES=24`
- 支持问题驱动选帧和长视频候选时间窗二次加密

---

## Benchmark

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

# 问题驱动选帧 + 稳定上下文
python -m src.benchmark.run_videomme --durations short --max-frames 96 --question-topk-frames 24 --question-neighbor-window 1
```

更多命令见：

- [src/benchmark/commands.md](./src/benchmark/commands.md)
- [commands.md](./commands.md)

---

## 项目结构

```text
Video-RAG-Ultra/
├── src/
│   ├── app.py
│   ├── video_processor.py
│   ├── audio_processor.py
│   ├── vlm_handler.py
│   ├── vision_backends.py
│   ├── clip_demo.py
│   ├── video_retriever.py
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

### 核心模块说明

- [src/app.py](./src/app.py)：Gradio UI、前端交互与服务编排
- [src/video_processor.py](./src/video_processor.py)：双路采样、视觉编码与检索
- [src/audio_processor.py](./src/audio_processor.py)：音频提取、转录、文本向量检索
- [src/vlm_handler.py](./src/vlm_handler.py)：Qwen2.5-VL 加载、event chunk 组织与回答生成
- [src/vision_backends.py](./src/vision_backends.py)：OpenCLIP / CLIP 后端适配层
- [src/benchmark/](./src/benchmark)：Video-MME 数据、推理和评测

---

## 配置说明

### 视觉相关

- `VIDEO_RETRIEVER_BACKEND`：默认 `openclip`
- `VIDEO_RETRIEVER_MODEL`：默认 `ViT-L-14`
- `VIDEO_RETRIEVER_PRETRAINED`：默认 `openai`
- `VIDEO_SCENE_SAMPLE_FPS`：场景候选采样频率，默认 `2.0`

### 音频相关

- `AUDIO_ASR_BACKEND`：默认 `faster-whisper`
- `AUDIO_ASR_MODEL`：默认 `large-v3`
- `AUDIO_ASR_USE_VAD`：默认 `1`
- `AUDIO_TEXT_EMBEDDING_MODEL`：默认 `BAAI/bge-m3`

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

---

## 常见问题

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

---

## 致谢

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Gradio](https://github.com/gradio-app/gradio)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 许可证

本项目采用 MIT 许可证。
