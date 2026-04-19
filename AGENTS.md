# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/`. Use `src/app.py` for the Gradio entrypoint, `src/video_processor.py` for visual retrieval, `src/audio_processor.py` for ASR and text retrieval, and `src/vlm_handler.py` for Qwen2.5-VL inference. Supporting docs are in `README.md`, `commands.md`, and `PROJECT_STRUCTURE.md`. Sample and generated data belong under `data/videos/` and `data/embeddings/`; keep large models, cached outputs, keyframes, logs, and `.gradio/` artifacts out of commits.

## Build, Test, and Development Commands
There is no packaging or build step yet; work from a Python 3.10+ virtual environment.

```bash
pip install -r requirements.txt
python src/app.py
python src/clip_demo.py
python -c "from src.video_processor import VideoRetriever; VideoRetriever()"
python -c "from src.audio_processor import AudioRetriever; AudioRetriever()"
python -c "from src.vlm_handler import VLMHandler; print(VLMHandler(max_retries=1).available)"
```

Use the first command to install dependencies, the second to launch the UI, and the remaining commands for targeted smoke checks on major modules.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise docstrings on non-obvious methods. Keep modules focused by responsibility instead of adding large multi-purpose files. Prefer explicit environment-variable reads for runtime configuration and keep user-facing strings consistent with the current bilingual Chinese/English style. No formatter is enforced in-repo; if you use one locally, avoid large unrelated reformatting.

## Testing Guidelines
No formal `tests/` suite or coverage gate is checked in today. For every change, run at least the relevant smoke check above and verify `python src/app.py` still starts cleanly. If you add automated tests, place them in a new `tests/` directory and name files `test_*.py` so a future `pytest` setup can adopt them without churn.

## Commit & Pull Request Guidelines
Recent history favors short imperative commits, usually with Conventional Commit prefixes such as `feat:`, `fix:`, and `refactor:`. Keep subjects focused on one change. PRs should include a clear summary, note any model or environment-variable changes, link the related issue when available, and attach screenshots or short recordings for UI updates in `src/app.py`.
