# Video-RAG Ultra: 多模态视频理解系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**基于检索增强生成 (RAG) 技术的超长视频问答系统**

*视觉 + 音频双模态检索 · 多GPU加速 · 实时交互*

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [技术架构](#-技术架构) • [使用示例](#-使用示例)

</div>

---

## 📖 项目介绍

Video-RAG Ultra 是一个基于 RAG (Retrieval-Augmented Generation) 技术的长视频问答系统。它通过**视觉关键帧检索**和**音频语义检索**的双模态索引，结合 **Qwen2.5-VL** 视觉语言模型，实现对超长视频内容的智能问答。

### 核心能力

- 🎥 **视觉理解**：基于 CLIP 的关键帧语义检索，精准定位视频画面
- 🎤 **音频理解**：Whisper 语音转录 + 文本语义检索，理解视频对话内容
- 🧩 **音频分段与缓存**：超长音频分段转录，自动缓存转录结果，加速二次构建
- 🤖 **智能问答**：多模态证据融合，生成可解释的答案
- ⚡ **多GPU加速**：模型分散加载，充分利用多卡资源
- 🎨 **精美界面**：现代化 Web UI，实时展示检索证据

---

## ✨ 功能特性

### 多模态检索

- **视觉检索**：使用 CLIP (ViT-B/32) 对视频关键帧进行向量化，支持文本查询检索相关画面
- **音频检索**：使用 Whisper 进行语音转录，结合 Sentence-Transformer 进行语义检索
- **分段转录**：默认按 300 秒分段，避免超长视频转录过慢或失败
- **结果缓存**：转录分段自动缓存，重复处理同一视频时显著加速
- **联合检索**：同时检索视觉和音频证据，提供更全面的上下文信息

### 智能问答

- **证据增强**：基于检索到的关键帧和音频片段生成答案
- **可解释性**：展示检索到的时间戳、相似度分数和关键帧画廊
- **多轮对话**：支持连续提问，保持对话上下文

### 性能优化

- **多GPU部署**：VLM、CLIP、音频模型分别部署在不同 GPU，避免显存溢出
- **流式处理**：支持超长视频，自动提取关键帧，减少计算量
- **实时反馈**：处理过程实时显示，用户体验友好

### 用户界面

- **现代化设计**：蓝紫渐变科技风，毛玻璃效果，流畅动画
- **交互友好**：一键上传、构建索引、实时问答
- **证据可视化**：关键帧画廊、时间戳标注、相似度展示

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐多 GPU 环境)
- ffmpeg (用于视频/音频处理)
- 至少 3 张 GPU (推荐 RTX 3090 或更高)

### 1. 创建 Conda 环境（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd Video-RAG-Ultra

# 创建并激活 Conda 环境
conda create -n video-rag-ultra python=3.10 -y
conda activate video-rag-ultra

# 如遇 defaults 源网络问题，可改用 conda-forge
# conda create -n video-rag-ultra python=3.10 -y -c conda-forge --override-channels

# 基础构建工具
python -m pip install -U pip setuptools wheel
```

### 2. 安装项目依赖

```bash
# 安装 PyTorch（GPU，CUDA 12.1）
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其余依赖（跳过 requirements.txt 中的 CLIP Git 依赖）
grep -v 'openai/CLIP.git' requirements.txt > /tmp/requirements.no_clip.txt
pip install -r /tmp/requirements.no_clip.txt

# Qwen2.5-VL 需要较新版本（见 src/vlm_handler.py）
pip install "transformers>=4.49.0,<5.0.0"
```

### 3. 安装 CLIP（在线/离线二选一）

```bash
# 在线安装（网络可访问 GitHub 时）
pip install git+https://github.com/openai/CLIP.git
```

```bash
# 离线安装（已下载源码到 ./CLIP-main）
conda install -y "setuptools<81" wheel
pip install --no-build-isolation -e ./CLIP-main
```

### 4. 设置环境变量（可选）

```bash
# 若出现 Connection reset by peer，可先清除代理配置
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

# HuggingFace 镜像（国内推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 可见 GPU（按机器实际情况设置）
export CUDA_VISIBLE_DEVICES=0,1,2
```

### 5. 验证环境

```bash
# 检查 CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

# 验证 CLIP 环境
python src/clip_demo.py

# 验证关键依赖
python -c "import clip, whisper, faiss, transformers, gradio; print('All dependencies OK')"
```

### 6. 启动应用

```bash
HF_ENDPOINT=https://hf-mirror.com python src/app.py
```

应用将在 `http://0.0.0.0:7860` 启动，Gradio 会自动生成公网链接。

**GPU 注意**：当前代码默认优先使用 `cuda:1`（可通过 `VLM_DEVICE` 覆盖），建议至少使用 2 张可见 GPU。单卡环境可设置 `VLM_DEVICE=cuda:0`。

### 7. 使用流程

1. **上传视频**：在左侧控制面板上传视频文件
2. **构建索引**：点击"🚀 构建索引"按钮，系统将自动：
   - 提取视频关键帧并向量化
   - 转录音频并构建文本索引
3. **开始问答**：在右侧输入框提问，例如：
   - "老师讲了哪三个核心概念？"
   - "视频中出现了哪些场景？"
   - "这个视频的主要内容是什么？"

---

## 🏗️ 技术架构

### 系统架构

```css
┌─────────────────────────────────────────────────────────────────────┐
│                          Video-RAG Ultra                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────────── Offline Indexing ────────────────────┐ │
│   │                                                               │ │
│   │   ┌──────────────┐                                           │ │
│   │   │   视频上传    │                                           │ │
│   │   └──────┬───────┘                                           │ │
│   │          │                                                   │ │
│   │   ┌──────▼──────┐        ┌──────────────┐                   │ │
│   │   │ 关键帧提取   │──────▶│  CLIP 编码     │                   │ │
│   │   │ (OpenCV)    │        │ (ViT-B/32)    │                   │ │
│   │   └──────┬──────┘        └──────┬───────┘                   │ │
│   │          │                      │                           │ │
│   │          │               ┌──────▼──────┐                    │ │
│   │          │               │ Visual Index │                    │ │
│   │          │               │  FAISS.add() │                    │ │
│   │          │               └──────────────┘                    │ │
│   │                                                                  │
│   │   ┌──────────────┐        ┌──────────────┐                   │ │
│   │   │ 音频提取      │──────▶│ Whisper 转录   │                   │ │
│   │   │ (FFmpeg)     │        │ (large-v3)    │                   │ │
│   │   └──────┬──────┘        └──────┬───────┘                   │ │
│   │          │                      │                           │ │
│   │          │               ┌──────▼──────┐                    │ │
│   │          │               │ Sentence-T   │                    │ │
│   │          │               │ (Text Embed) │                    │ │
│   │          │               └──────┬───────┘                    │ │
│   │          │                      │                           │ │
│   │          │               ┌──────▼──────┐                    │ │
│   │          │               │  Audio Index │                    │ │
│   │          │               │  FAISS.add() │                    │ │
│   │          │               └──────────────┘                    │ │
│   │                                                                  │
│   └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
│────────────────────────── Online Inference ─────────────────────────│
│                                                                     │
│   ┌──────────────┐                                                │
│   │   用户提问   │                                                │
│   └──────┬───────┘                                                │
│          │                                                        │
│   ┌──────▼──────┐                                                │
│   │ Query 编码   │                                                │
│   │ (CLIP / ST) │                                                │
│   └──────┬──────┘                                                │
│          │                                                        │
│   ┌──────▼──────────────────────────────────┐                    │
│   │           Multi-Modal Retrieval          │                    │
│   │                                          │                    │
│   │   ┌──────────────┐   ┌──────────────┐   │                    │
│   │   │ Visual Search │   │  Audio Search │   │                    │
│   │   │ FAISS.search │   │ FAISS.search │   │                    │
│   │   └──────┬───────┘   └──────┬───────┘   │                    │
│   │          │                  │           │                    │
│   │   关键帧 + 时间戳      文本 + 时间戳     │                    │
│   └──────────┬──────────────────┬──────────┘                    │
│              │                  │                               │
│              └──────────┬───────┘                               │
│                         ▼                                       │
│        ┌──────────────────────────────────────┐                │
│        │     Qwen2.5-VL-7B-Instruct (VLM)       │                │
│        │  多模态证据融合 + 推理 + 答案生成       │                │
│        └──────────┬───────────────────────────┘                │
│                   ▼                                              │
│               最终回答                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

```





### 技术栈

| 模块             | 技术                 | 说明                                  |
| ---------------- | -------------------- | ------------------------------------- |
| **视觉编码**     | CLIP (ViT-B/32)      | OpenAI CLIP 模型，512维向量           |
| **音频转录**     | Whisper large-v3     | OpenAI Whisper 模型，高精度语音识别   |
| **文本编码**     | Sentence-Transformer | all-MiniLM-L6-v2，384维向量           |
| **向量检索**     | FAISS                | Facebook AI Similarity Search，L2距离 |
| **视觉语言模型** | Qwen2.5-VL-7B-Instruct | 阿里通义千问视觉语言模型            |
| **Web框架**      | Gradio               | 快速构建交互式界面                    |
| **视频处理**     | OpenCV + FFmpeg      | 关键帧提取、音频提取                  |

### GPU 分配策略

- **GPU 0 (cuda:0)**：CLIP 模型（视觉编码）
- **GPU 1 (cuda:1)**：Qwen2.5-VL（视觉语言模型）
- **GPU 2 (cuda:2)**：Whisper + Sentence-Transformer（音频处理）

---

## 📝 使用示例

### 示例 1：教育视频问答

**用户提问**："老师讲了哪三个核心概念？"

**系统响应**：

- 🔍 **RAG 多模态证据**
  - 👁️ **视觉证据**：显示相关关键帧的时间戳和相似度
  - 👂 **音频证据**：显示相关语音片段的时间戳和文本
- 🤖 **AI 分析结果**：基于检索证据生成的详细答案

### 示例 2：场景识别

**用户提问**："视频中出现了哪些场景？"

系统会检索并展示：

- 不同场景的关键帧
- 场景切换的时间点
- 每个场景的描述

---

## 📁 项目结构

```
Video-RAG/
├── src/
│   ├── app.py                 # Gradio Web 应用主程序
│   ├── video_processor.py     # 视频关键帧提取与检索
│   ├── audio_processor.py     # 音频转录与检索
│   ├── vlm_handler.py         # Qwen2.5-VL 模型处理
│   ├── clip_demo.py           # CLIP 环境验证脚本
│   └── keyframes/             # 关键帧存储目录
├── data/
│   └── videos/                # 视频文件目录
├── requirements.txt           # Python 依赖
└── README.md                  # 项目文档
```

---

## 🔧 配置说明

### 环境变量

```bash
# HuggingFace 镜像（可选，加速模型下载）
export HF_ENDPOINT=https://hf-mirror.com

# CUDA 设备（可选，默认自动分配）
export CUDA_VISIBLE_DEVICES=0,1,2

# （可选）音频转录缓存目录（默认：../data/embeddings/audio_cache）
# export AUDIO_CACHE_DIR=../data/embeddings/audio_cache
```

### 模型配置

- **CLIP 模型**：默认使用 `ViT-B/32`，可在 `VideoRetriever` 初始化时修改
- **Whisper 模型**：默认使用 `medium`，可在 `AudioRetriever` 中修改
- **Qwen2.5-VL 模型**：默认从 HuggingFace 下载，支持本地路径

### 音频分段与缓存配置

`AudioRetriever` 支持分段时长与缓存目录配置：

- `chunk_seconds`：分段时长（秒），默认 300
- `cache_dir`：转录缓存目录，默认 `../data/embeddings/audio_cache`

示例：

- `AudioRetriever(chunk_seconds=300, cache_dir="../data/embeddings/audio_cache")`

---

## 🎯 项目进度

### ✅ 已完成

- [x] 项目结构搭建
- [x] 环境依赖定义
- [x] 关键帧提取与去重
- [x] CLIP 向量化与 FAISS 索引
- [x] Whisper 音频转录
- [x] 文本语义检索
- [x] Qwen2.5-VL 模型集成
- [x] 多模态 RAG 问答
- [x] Gradio Web 界面开发
- [x] 多GPU 部署优化
- [x] UI 美化与交互优化

### 🚧 待优化

- [ ] 支持更多视频格式
- [ ] 索引持久化存储
- [ ] 批量视频处理
- [ ] 性能监控与日志
- [ ] API 接口开发

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目采用 MIT 许可证。

---

## 🙏 致谢

- [OpenAI CLIP](https://github.com/openai/CLIP) - 视觉编码模型
- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别模型
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - 视觉语言模型
- [Gradio](https://github.com/gradio-app/gradio) - Web 界面框架
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索库

---

<div align="center">


**⭐ 如果这个项目对你有帮助，请给个 Star！**

Made with ❤️ by Video-RAG Team

</div>
