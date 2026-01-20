# 常用命令参考

## 📋 快速命令索引

### 环境设置
- [安装依赖](#安装依赖)
- [设置环境变量](#设置环境变量)
- [验证环境](#验证环境)

### 运行应用
- [启动应用](#启动应用)
- [后台运行](#后台运行)
- [指定端口](#指定端口)

### 系统检查
- [检查 GPU](#检查-gpu)
- [检查依赖](#检查依赖)
- [清理临时文件](#清理临时文件)

---

## 🔧 环境设置

### 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 如果使用 GPU，安装 faiss-gpu（可选）
pip install faiss-gpu>=1.7.4

# 安装额外依赖（如果 requirements.txt 中未包含）
pip install openai-whisper sentence-transformers
```

### 设置环境变量

```bash
# 设置 HuggingFace 镜像（国内用户推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 设置 CUDA 设备（可选，默认自动分配）
export CUDA_VISIBLE_DEVICES=0,1,2

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 验证环境

```bash
# 验证 CLIP 环境
python src/clip_demo.py

# 验证 Python 版本
python --version

# 验证 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 验证所有关键依赖
python -c "import gradio, clip, whisper, faiss, transformers; print('All dependencies OK')"
```

---

## 🚀 启动应用

### 基本启动

```bash
# 进入源码目录
cd src

# 启动应用（使用 HuggingFace 镜像）
HF_ENDPOINT=https://hf-mirror.com python3 app.py
```

### 指定端口

```bash
# 修改 app.py 中的端口号，或使用环境变量
PORT=8080 python3 app.py
```

### 后台运行

```bash
# 使用 nohup 后台运行
cd src
nohup HF_ENDPOINT=https://hf-mirror.com python3 app.py > ../logs/app.log 2>&1 &

# 查看进程
ps aux | grep app.py

# 查看日志
tail -f logs/app.log

# 停止后台进程
pkill -f app.py
```

### 使用 screen（推荐）

```bash
# 创建新的 screen 会话
screen -S video-rag

# 启动应用
cd src
HF_ENDPOINT=https://hf-mirror.com python3 app.py

# 分离会话：按 Ctrl+A，然后按 D
# 重新连接：screen -r video-rag
# 列出所有会话：screen -ls
```

### 使用 tmux

```bash
# 创建新的 tmux 会话
tmux new -s video-rag

# 启动应用
cd src
HF_ENDPOINT=https://hf-mirror.com python3 app.py

# 分离会话：按 Ctrl+B，然后按 D
# 重新连接：tmux attach -t video-rag
# 列出所有会话：tmux ls
```

---

## 🔍 系统检查

### 检查 GPU

```bash
# 查看 GPU 信息
nvidia-smi

# 查看 GPU 使用情况（实时监控）
watch -n 1 nvidia-smi

# 检查 PyTorch 是否能使用 GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 检查依赖

```bash
# 列出已安装的包
pip list

# 检查特定包版本
pip show torch
pip show gradio
pip show transformers

# 检查依赖冲突
pip check
```

### 清理临时文件

```bash
# 清理 Python 缓存
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# 清理关键帧目录
rm -rf src/keyframes/*

# 清理临时音频文件
find . -name "*.wav" -delete
find . -name "*.m4a" -delete

# 清理转码视频
find . -name "*_h264.mp4" -delete

# 清理所有临时文件（谨慎使用）
find . -name "*.tmp" -delete
find . -name "*.temp" -delete
```

---

## 🛠️ 开发调试

### 运行单个模块测试

```bash
# 测试视频处理
python -c "from src.video_processor import VideoRetriever; r = VideoRetriever(); print('VideoRetriever OK')"

# 测试音频处理
python -c "from src.audio_processor import AudioRetriever; a = AudioRetriever(); print('AudioRetriever OK')"

# 测试 VLM（需要较长时间）
python -c "from src.vlm_handler import VLMHandler; v = VLMHandler(); print('VLMHandler OK')"
```

### 查看日志

```bash
# 如果使用后台运行，查看日志
tail -f logs/app.log

# 查看最近的错误
grep -i error logs/app.log | tail -20

# 查看模型加载日志
grep -i "loading\|init" logs/app.log
```

---

## 📦 项目维护

### 更新依赖

```bash
# 更新所有包到最新版本（谨慎使用）
pip install --upgrade -r requirements.txt

# 更新特定包
pip install --upgrade gradio
pip install --upgrade transformers
```

### 备份重要文件

```bash
# 备份配置文件
cp requirements.txt requirements.txt.bak
cp .gitignore .gitignore.bak

# 备份代码（创建压缩包）
tar -czf video-rag-backup-$(date +%Y%m%d).tar.gz src/ requirements.txt README.md
```

### 检查代码质量

```bash
# 使用 flake8 检查代码风格（如果安装了）
flake8 src/

# 使用 pylint 检查代码（如果安装了）
pylint src/
```

---

## 🐛 故障排查

### 常见问题

```bash
# 问题：端口被占用
# 解决：查找占用端口的进程
lsof -i :7860
# 或
netstat -tulpn | grep 7860
# 杀死进程
kill -9 <PID>

# 问题：显存不足
# 解决：检查 GPU 使用情况
nvidia-smi
# 释放显存：重启 Python 进程

# 问题：模型下载失败
# 解决：使用镜像或手动下载
export HF_ENDPOINT=https://hf-mirror.com
# 或设置代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 问题：ffmpeg 未找到
# 解决：安装 ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
# 验证
ffmpeg -version
```

---

## 📝 快速参考

### 最常用命令

```bash
# 一键启动（推荐）
cd src && HF_ENDPOINT=https://hf-mirror.com python3 app.py

# 后台启动
cd src && nohup HF_ENDPOINT=https://hf-mirror.com python3 app.py > ../logs/app.log 2>&1 &

# 清理缓存
find . -type d -name __pycache__ -exec rm -r {} + && find . -name "*.pyc" -delete
```

### 环境检查清单

```bash
# 快速检查所有环境
echo "=== Python ===" && python --version
echo "=== CUDA ===" && python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo "=== FFmpeg ===" && ffmpeg -version | head -1
echo "=== Dependencies ===" && python -c "import gradio, clip, whisper, faiss; print('OK')"
```

---

## 💡 提示

- **首次运行**：模型下载可能需要较长时间，请耐心等待
- **多 GPU 环境**：确保至少有 3 张 GPU 以获得最佳性能
- **显存管理**：如果显存不足，可以降低模型大小（如使用 Whisper medium 而非 large-v3）
- **网络问题**：国内用户强烈建议使用 HuggingFace 镜像
- **后台运行**：推荐使用 screen 或 tmux，便于管理和查看日志

---

**最后更新**: 2026-12-20
