# 项目结构说明

## 目录结构

```
Video-RAG/
├── .github/                    # GitHub 配置
│   └── ISSUE_TEMPLATE/        # Issue 模板
├── .gitignore                 # Git 忽略文件
├── LICENSE                    # MIT 许可证
├── README.md                  # 项目文档
├── requirements.txt           # Python 依赖
├── commands.md                # 快速命令参考
├── PROJECT_STRUCTURE.md       # 本文件
│
├── src/                       # 源代码目录
│   ├── __init__.py           # Python 包初始化
│   ├── app.py                # Gradio Web 应用主程序
│   ├── video_processor.py    # 视频关键帧提取与检索
│   ├── audio_processor.py    # 音频转录与检索
│   ├── vlm_handler.py        # Qwen-VL 模型处理
│   ├── clip_demo.py          # CLIP 环境验证脚本
│   ├── video_retriever.py    # 视频检索演示（可选）
│   ├── keyframes/            # 关键帧存储（运行时生成）
│   └── SimSun.ttf            # 字体文件（如需要）
│
├── data/                      # 数据目录
│   ├── videos/               # 视频文件目录
│   └── embeddings/           # 向量索引存储（可选）
│
├── notebooks/                 # Jupyter  notebooks（可选）
│
└── CLIP-main/                # CLIP 子模块（应被 .gitignore 忽略）

```

## 文件说明

### 核心文件
- ✅ `README.md` - 完整的项目文档和使用说明
- ✅ `requirements.txt` - Python 依赖列表
- ✅ `LICENSE` - MIT 许可证
- ✅ `.gitignore` - Git 忽略规则

### 源代码
- ✅ `src/app.py` - 主应用程序（Gradio Web 界面）
- ✅ `src/video_processor.py` - 视频处理模块
- ✅ `src/audio_processor.py` - 音频处理模块
- ✅ `src/vlm_handler.py` - 视觉语言模型处理
- ✅ `src/clip_demo.py` - CLIP 验证脚本
- ✅ `src/__init__.py` - Python 包初始化

### 配置文件
- ✅ `.github/ISSUE_TEMPLATE/` - Issue 模板
- ✅ `commands.md` - 快速命令参考

### 数据目录
- ✅ `data/videos/` - 视频文件存储
- ✅ `data/embeddings/` - 向量索引存储

## 注意事项

1. **CLIP-main 目录**: 这是 CLIP 的源代码，已添加到 `.gitignore`，不应该提交到仓库
2. **keyframes 目录**: 运行时生成，会被 `.gitignore` 忽略
3. **模型文件**: 大型模型文件（.pt, .pth, .bin）会被忽略，用户需要自己下载

## 建议

如果需要更完善的结构，可以考虑添加：
- `CONTRIBUTING.md` - 贡献指南
- `CHANGELOG.md` - 版本更新日志
- `setup.py` 或 `pyproject.toml` - 用于打包发布
- `tests/` - 单元测试
- `docs/` - 详细文档
