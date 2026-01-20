import gradio as gr
import os
import traceback
from video_processor import VideoRetriever
from vlm_handler import VLMHandler
from audio_processor import AudioRetriever

print("正在初始化 Web 系统 (这可能需要加载多个模型)...")

try:
    vlm = VLMHandler()
    retriever = VideoRetriever()
    audio_retriever = AudioRetriever()
except Exception as e:
    print(f"模型加载出错: {e}")
    raise 

def process_upload(video_path):
    """Process video: visual and audio indexing"""
    if video_path is None: return "请上传视频", None
    
    loading_html = (
        f"<div class='loading-pane'>"
        f"<div class='spinner'></div>"
        f"<div class='loading-text'>正在处理 <b>{os.path.basename(video_path)}</b>...</div>"
        f"<div class='loading-subtext'>提取视觉关键帧中，请稍候</div>"
        f"</div>"
    )
    yield loading_html, None
    
    try:
        retriever.process_video(video_path, max_duration_minutes=None)
        loading_html = (
            f"<div class='loading-pane'>"
            f"<div class='spinner'></div>"
            f"<div class='loading-text'>正在进行音频转录</div>"
            f"<div class='loading-subtext'>使用 Whisper Large-v3 模型处理中...</div>"
            f"</div>"
        )
        yield loading_html, None
        audio_retriever.process_audio(video_path)
        stats = (
            f"<div class='success-pane'>"
            f"<div class='success-header'>"
            f"<div class='check-icon'></div>"
            f"<span class='success-title'>索引构建完成！</span>"
            f"</div>"
            f"<div class='stats-grid'>"
            f"<div class='stat-item'><span class='stat-label'>视觉关键帧</span><span class='stat-value'>{retriever.index.ntotal}</span><span class='stat-unit'>帧</span></div>"
            f"<div class='stat-item'><span class='stat-label'>音频片段</span><span class='stat-value'>{audio_retriever.index.ntotal}</span><span class='stat-unit'>条</span></div>"
            f"</div>"
            f"<div class='ready-badge'><span class='ready-icon'>✨</span> Ready to Chat!</div>"
            f"</div>"
        )
        yield stats, None
    except Exception as e:
        traceback.print_exc()
        yield f"<div class='error-pane'><span class='error-icon'>❌</span> 处理失败: <code>{str(e)}</code></div>", None

def chat_engine(query, history):
    """Core Q&A logic with multimodal retrieval"""
    if retriever.index.ntotal == 0:
        return "<div class='warn-pane'>⚠️ 请先在左侧上传视频并点击 [构建索引]</div>", []
    print(f"[App] Visual Search...")
    visual_results = retriever.search(query, k=6)
    print(f"[App] Audio Search...")
    audio_results = audio_retriever.search(query, k=6)
    
    gallery_data, images_info = [], []
    rag_evidence = (
        "<div class='rag-evidence-container'>"
        "<div class='rag-evidence-title'>🔍 RAG 多模态证据</div>"
        "<div class='evidence-block'>"
        "<div class='evidence-header'><span class='icon-eye'>👁️</span> <b>视觉证据</b> <span class='evidence-count'>({})</span></div>"
        "<ul class='evidence-list'>"
    ).format(len(visual_results))
    for ts, score, path in visual_results:
        time_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"
        gallery_data.append((path, f"Time: {time_str}"))
        images_info.append((ts, score, path))
        rag_evidence += (
            f"<li class='evidence-item'>"
            f"<span class='timestamp'>{time_str}</span>"
            f"<span class='sim-score'>相似度: {score:.3f}</span>"
            f"</li>"
        )
    rag_evidence += (
        "</ul></div>"
        "<div class='evidence-block'>"
        "<div class='evidence-header'><span class='icon-ear'>👂</span> <b>音频证据</b> <span class='evidence-count'>({})</span></div>"
        "<ul class='evidence-list'>"
    ).format(len(audio_results))
    for start, text, score in audio_results:
        time_str = f"{int(start)//60:02d}:{int(start)%60:02d}"
        rag_evidence += (
            f"<li class='evidence-item'>"
            f"<span class='timestamp'>{time_str}</span>"
            f"<span class='aud-text'>{text[:80]}{'...' if len(text) > 80 else ''}</span>"
            f"</li>"
        )
    rag_evidence += "</ul></div></div>"
    answer = vlm.chat(query, images_info, audio_results)
    final_response = (
        f"{rag_evidence}<div class='divider'></div>"
        f"<div class='ai-answer-title'>🤖 AI 分析结果</div>"
        f"<div class='ai-answer-block'>{answer}</div>"
    )
    return final_response, gallery_data

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');
:root {
    --blue: #6366f1;
    --purple: #a21caf;
    --bg-grad: linear-gradient(135deg, #0f172a 0%, #1e1b4b 25%, #312e81 50%, #4c1d95 75%, #6b21a8 100%);
    --bg-grad-animated: linear-gradient(135deg, #0f172a 0%, #1e1b4b 25%, #312e81 50%, #4c1d95 75%, #6b21a8 100%);
    --card-bg: rgba(250,250,255,0.75);
    --sidebar-bg: rgba(238,242,255,0.95);
    --accent: linear-gradient(135deg, #60a5fa 0%, #818cf8 50%, #a78bfa 100%);
    --accent-hover: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%);
    --shadow: 0 20px 60px rgba(99, 102, 241, 0.15), 0 8px 25px rgba(139, 92, 246, 0.1);
    --shadow-hover: 0 25px 80px rgba(99, 102, 241, 0.25), 0 10px 30px rgba(139, 92, 246, 0.15);
}
* {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
body, html {
    min-height: 100vh !important;
    min-width: 100vw;
    background: var(--bg-grad) fixed;
    background-size: 200% 200%;
    animation: gradientShift 15s ease infinite;
    font-family: 'Inter', 'Montserrat', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1e293b;
    letter-spacing: 0.01em;
    overflow-x: hidden;
}
@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.gradio-container {
    background: transparent !important;
    padding: 0 !important;
}
.container {
    max-width: 1400px; 
    margin: 0 auto; 
    padding: 2.5rem 1.5rem; 
    position: relative;
}
.header-text { 
    text-align: center; 
    margin-bottom: 3rem;
    animation: fadeInDown 0.8s ease-out;
}
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.header-text h1 {
    font-size: 3.2rem;
    letter-spacing: -0.02em;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a21caf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 40px rgba(139, 92, 246, 0.3);
    animation: titleGlow 3s ease-in-out infinite;
    position: relative;
}
@keyframes titleGlow {
    0%, 100% { filter: brightness(1) drop-shadow(0 0 20px rgba(139, 92, 246, 0.3)); }
    50% { filter: brightness(1.1) drop-shadow(0 0 30px rgba(139, 92, 246, 0.5)); }
}
.header-text p {
    color: #a78bfa;
    font-weight: 600;
    opacity: 0.95;
    font-size: 1.25em;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}
.sidebar-card {
    background: var(--sidebar-bg);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    box-shadow: var(--shadow);
    padding: 2rem 1.5rem;
    position: relative;
    border: 1px solid rgba(139, 92, 246, 0.2);
    min-height: 650px;
    animation: slideInLeft 0.6s ease-out;
}
@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
.sidebar-card::before {
    content: "";
    position: absolute;
    pointer-events: none;
    inset: -3px;
    border-radius: 26px;
    border: 2px solid;
    border-image: linear-gradient(135deg, rgba(99, 102, 241, 0.4), rgba(167, 139, 250, 0.4)) 1;
    opacity: 0.6;
}
.sidebar-card::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--accent);
    border-radius: 24px 24px 0 0;
    opacity: 0.8;
}
.gradio-accordion { 
    border-radius: 16px !important; 
    border: 1px solid rgba(139, 92, 246, 0.15) !important; 
    background: rgba(243, 244, 255, 0.8) !important;
    backdrop-filter: blur(10px);
}
#status-markdown {
    background: linear-gradient(135deg, rgba(240, 242, 255, 0.9), rgba(250, 250, 255, 0.9)) !important;
    color: #5b21b6 !important;
    padding: 1.2rem 1.5rem;
    font-size: 1.05em;
    border-radius: 16px;
    border: 1.5px solid rgba(139, 92, 246, 0.2);
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(167, 139, 250, 0.15);
    backdrop-filter: blur(10px);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 8px 24px rgba(167, 139, 250, 0.15); }
    50% { box-shadow: 0 8px 32px rgba(167, 139, 250, 0.25); }
}
#chatbot {
    min-height: 680px;
    border-radius: 24px;
    border: 1px solid rgba(139, 92, 246, 0.15);
    background: linear-gradient(135deg, rgba(252, 253, 255, 0.95), rgba(248, 250, 252, 0.95));
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow);
    padding: 1.5rem;
    animation: slideInRight 0.6s ease-out;
}
@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
.gradio-gallery { 
    background: transparent !important; 
    gap: 16px !important; 
    padding: 1rem 0;
}
.gradio-gallery img { 
    border-radius: 16px; 
    border: 2px solid rgba(139, 92, 246, 0.3); 
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.2);
    transition: all 0.3s ease;
    cursor: pointer;
}
.gradio-gallery img:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 16px 48px rgba(99, 102, 241, 0.35);
    border-color: rgba(139, 92, 246, 0.5);
}
.primary-btn {
    background: var(--accent) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05em;
    letter-spacing: 0.02em;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4), 0 4px 12px rgba(139, 92, 246, 0.3);
    border-radius: 16px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.primary-btn::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s;
}
.primary-btn:hover::before {
    left: 100%;
}
.primary-btn:hover { 
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.5), 0 6px 16px rgba(139, 92, 246, 0.4);
    background: var(--accent-hover) !important;
}
.primary-btn:active {
    transform: translateY(0) scale(0.98);
}
input, textarea, .gradio-textbox textarea {
    border-radius: 16px !important;
    border: 2px solid rgba(139, 92, 246, 0.3) !important;
    background: rgba(246, 248, 255, 0.95) !important;
    font-size: 1.05em !important;
    padding: 0.875rem 1.25rem !important;
    transition: all 0.3s ease;
}
input:focus, textarea:focus, .gradio-textbox textarea:focus {
    border-color: rgba(139, 92, 246, 0.6) !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1) !important;
    background: rgba(255, 255, 255, 1) !important;
}
.footer-text {
    color: #a78bfa;
    text-align: center;
    opacity: 0.9;
    font-size: 1em;
    margin-top: 1rem;
    letter-spacing: 0.03em;
    font-weight: 600;
}
/* 加载动画 */
.loading-pane {
    padding: 2rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(240, 242, 255, 0.95), rgba(250, 250, 255, 0.95));
    border-radius: 16px;
    border: 1.5px solid rgba(139, 92, 246, 0.2);
    box-shadow: 0 8px 24px rgba(167, 139, 250, 0.15);
}
.spinner {
    width: 48px;
    height: 48px;
    margin: 0 auto 1.5rem;
    border: 4px solid rgba(139, 92, 246, 0.2);
    border-top-color: #6366f1;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.loading-text {
    font-size: 1.15em;
    font-weight: 700;
    color: #4c1d95;
    margin-bottom: 0.5rem;
}
.loading-subtext {
    font-size: 0.95em;
    color: #7c3aed;
    opacity: 0.8;
}
/* 成功面板优化 */
.success-pane {
    padding: 2rem;
    background: linear-gradient(135deg, rgba(236, 253, 245, 0.95), rgba(240, 253, 250, 0.95));
    border-radius: 18px;
    box-shadow: 0 12px 32px rgba(34, 197, 94, 0.15);
    border: 2px solid rgba(34, 197, 94, 0.3);
    margin: 1rem 0;
    animation: successSlideIn 0.5s ease-out;
}
@keyframes successSlideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.success-header {
    display: flex;
    align-items: center;
    margin-bottom: 1.5rem;
    gap: 0.75rem;
}
.check-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    border-radius: 50%;
    color: white;
    font-size: 1.2em;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
    animation: checkPop 0.5s ease-out;
}
@keyframes checkPop {
    0% { transform: scale(0); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
.success-title {
    font-size: 1.3em;
    font-weight: 800;
    color: #15803d;
    font-family: 'Montserrat', sans-serif;
}
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-item {
    background: rgba(255, 255, 255, 0.8);
    padding: 1.25rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(34, 197, 94, 0.2);
    transition: all 0.3s ease;
}
.stat-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(34, 197, 94, 0.2);
}
.stat-label {
    display: block;
    font-size: 0.9em;
    color: #64748b;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.stat-value {
    display: block;
    font-size: 2em;
    font-weight: 800;
    color: #22c55e;
    font-family: 'Montserrat', sans-serif;
    line-height: 1;
}
.stat-unit {
    font-size: 0.85em;
    color: #64748b;
    margin-left: 0.25rem;
    font-weight: 600;
}
.ready-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.05em;
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.3);
    animation: readyPulse 2s ease-in-out infinite;
}
@keyframes readyPulse {
    0%, 100% { transform: scale(1); box-shadow: 0 4px 16px rgba(34, 197, 94, 0.3); }
    50% { transform: scale(1.02); box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4); }
}
.ready-icon {
    font-size: 1.2em;
    animation: sparkle 1.5s ease-in-out infinite;
}
@keyframes sparkle {
    0%, 100% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(1.1); }
}
.error-pane {
    padding: 1.5rem;
    color: #dc2626;
    background: linear-gradient(135deg, rgba(254, 242, 242, 0.95), rgba(255, 235, 235, 0.95));
    border-radius: 16px;
    border: 2px solid rgba(239, 68, 68, 0.3);
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.15);
    font-weight: 700;
    animation: errorShake 0.5s ease-out;
}
@keyframes errorShake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}
.error-icon {
    margin-right: 0.5rem;
    font-size: 1.2em;
}
.error-pane code {
    background: rgba(239, 68, 68, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 0.9em;
}
.warn-pane {
    background: linear-gradient(135deg, rgba(254, 249, 195, 0.95), rgba(255, 251, 235, 0.95));
    border-radius: 14px;
    color: #b45309;
    font-size: 1.05em;
    padding: 1.25rem;
    border: 2px solid rgba(250, 204, 21, 0.3);
    margin-bottom: 0.75rem;
    font-weight: 700;
    box-shadow: 0 6px 20px rgba(250, 204, 21, 0.15);
}
.rag-evidence-container {
    margin-bottom: 1.5rem;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.2);
}
.rag-evidence-title {
    font-size: 1.25em;
    background: var(--accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0;
    font-family: 'Montserrat', sans-serif;
    position: relative;
    background-color: rgba(243, 244, 255, 0.5);
    backdrop-filter: blur(10px);
}
.rag-evidence-title::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.4;
}
.evidence-block {
    background: linear-gradient(135deg, rgba(243, 244, 255, 0.95), rgba(250, 250, 255, 0.95));
    padding: 1.25rem 1.5rem;
    border-top: 1px solid rgba(139, 92, 246, 0.15);
}
.evidence-block:last-child {
    border-radius: 0 0 16px 16px;
}
.evidence-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    font-size: 1.05em;
    color: #4c1d95;
    font-weight: 700;
}
.evidence-count {
    font-size: 0.85em;
    color: #7c3aed;
    font-weight: 600;
    margin-left: auto;
    background: rgba(139, 92, 246, 0.1);
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
}
.icon-eye, .icon-ear {
    font-weight: 900;
    font-size: 1.2em;
}
.evidence-list {
    padding-left: 0;
    margin: 0;
    list-style: none;
}
.evidence-item {
    margin-block: 0.75rem;
    padding: 0.875rem 1rem;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 10px;
    border-left: 3px solid rgba(139, 92, 246, 0.4);
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.evidence-item:hover {
    transform: translateX(4px);
    background: rgba(255, 255, 255, 0.9);
    border-left-color: rgba(139, 92, 246, 0.7);
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
}
.timestamp {
    background: linear-gradient(135deg, #ede9fe, #f3e8ff);
    color: #5b21b6;
    border-radius: 8px;
    padding: 0.4rem 0.85rem;
    font-weight: 700;
    font-size: 0.9em;
    border: 1px solid rgba(139, 92, 246, 0.25);
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    font-family: 'Monaco', 'Courier New', monospace;
    box-shadow: 0 2px 6px rgba(139, 92, 246, 0.1);
    transition: all 0.2s ease;
}
.timestamp:hover {
    background: linear-gradient(135deg, #ddd6fe, #e9d5ff);
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(139, 92, 246, 0.2);
}
.sim-score {
    color: #3b82f6;
    font-weight: 600;
    font-size: 0.9em;
    background: rgba(59, 130, 246, 0.1);
    padding: 0.3rem 0.7rem;
    border-radius: 8px;
    white-space: nowrap;
}
.aud-text {
    color: #4c1d95;
    font-weight: 500;
    line-height: 1.6;
    flex: 1;
    min-width: 200px;
}
.divider {
    height: 3px;
    width: 100%;
    margin: 1.5rem 0 1rem 0;
    border-radius: 2px;
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.4) 0%, rgba(167, 139, 250, 0.4) 50%, rgba(162, 28, 175, 0.3) 100%);
    position: relative;
    overflow: hidden;
}
.divider::after {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
    animation: shimmer 2s infinite;
}
@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}
.ai-answer-title {
    font-size: 1.25em;
    font-weight: 800;
    background: linear-gradient(135deg, #4c1d95, #6b21a8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
    font-family: 'Montserrat', sans-serif;
}
.ai-answer-block {
    background: linear-gradient(135deg, rgba(236, 233, 254, 0.85), rgba(243, 244, 255, 0.85));
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-block: 0.75rem;
    color: #1e293b;
    font-weight: 500;
    font-size: 1.08em;
    line-height: 1.8;
    border-left: 5px solid #818cf8;
    box-shadow: 0 6px 20px rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-left: 5px solid #818cf8;
}
/* 视频上传区域优化 */
.gradio-video {
    border-radius: 16px !important;
    overflow: hidden;
    border: 2px solid rgba(139, 92, 246, 0.2) !important;
    transition: all 0.3s ease;
}
.gradio-video:hover {
    border-color: rgba(139, 92, 246, 0.4) !important;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
}
/* 聊天消息优化 */
.message {
    animation: messageSlideIn 0.4s ease-out;
}
@keyframes messageSlideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
/* 额外样式优化 */
.section-title {
    color: #4c1d95 !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    font-family: 'Montserrat', sans-serif !important;
}
.gallery-accordion {
    margin-top: 1rem;
}
/* 响应式优化 */
@media (max-width: 768px) {
    .header-text h1 {
        font-size: 2rem;
    }
    .container {
        padding: 1.5rem 1rem;
    }
    .stats-grid {
        grid-template-columns: 1fr;
    }
    .sidebar-card {
        min-height: auto;
    }
    #chatbot {
        min-height: 500px;
        height: 500px;
    }
}
/* 滚动条美化 */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: rgba(243, 244, 255, 0.5);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    border-radius: 10px;
    border: 2px solid rgba(243, 244, 255, 0.5);
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="purple",
    spacing_size="md",
    radius_size="lg"
)

with gr.Blocks(title="Video-RAG Ultra | 多模态视频理解系统") as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header-text"):
            gr.Markdown(
                "<h1>⚡️ Video-RAG Ultra</h1>", 
                elem_id="main-title"
            )
            gr.Markdown(
                "<p style='margin-top: 0.5rem;'>让 AI 成为你的视频大脑</p>"
                "<p style='font-size: 1.1em; margin-top: 0.25rem;'>"
                "<span style='color:#a78bfa; font-weight: 600;'>超长视频理解 · 视觉 + 音频 检索增强生成 (RAG)</span>"
                "</p>",
                elem_id="subtitle"
            )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="sidebar-card"):
                gr.Markdown(
                    "### 🎛️ 控制中心",
                    elem_classes="section-title"
                )
                video_in = gr.Video(
                    label="🌟 上传视频", 
                    sources=["upload"], 
                    height=260, 
                    interactive=True
                )
                btn_process = gr.Button(
                    "🚀 构建索引", 
                    variant="primary", 
                    elem_classes="primary-btn",
                    scale=1
                )
                gr.Markdown("---")
                gr.Markdown(
                    "#### 📊 系统状态",
                    elem_classes="section-title"
                )
                status_display = gr.Markdown(
                    "<div style='text-align: center; padding: 1rem; color: #64748b;'>"
                    "⏸️ 等待视频上传...</div>", 
                    elem_id="status-markdown"
                )
                with gr.Accordion(
                    "🖼️ 检索关键帧画廊", 
                    open=False,
                    elem_classes="gallery-accordion"
                ):
                    gallery = gr.Gallery(
                        label="Visual Evidence", 
                        columns=3, 
                        height=360, 
                        show_label=False
                    )
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="💬 Qwen-VL-Chat (Audio-Enhanced)",
                    elem_id="chatbot",
                    height=700,
                    avatar_images=("👤", "🤖"),
                )
                with gr.Row():
                    msg = gr.Textbox(
                        show_label=False, 
                        placeholder="💡 请输入关于视频的提问，例如：'老师讲了哪三个核心概念？' 或 '视频中出现了哪些场景？'", 
                        scale=9,
                        container=False,
                        autofocus=True,
                        lines=2
                    )
                    btn_send = gr.Button(
                        "发送 ✨", 
                        variant="primary", 
                        scale=1, 
                        elem_classes="primary-btn",
                        size="lg"
                    )
                with gr.Row():
                    btn_clear = gr.Button(
                        "🗑️ 清空历史", 
                        size="sm", 
                        variant="secondary",
                        scale=1
                    )
                    gr.Markdown(
                        "<div style='text-align: right;'>"
                        "<span style='font-weight: 600; color: #a78bfa;'>"
                        "🚀 <b>Powered by</b> 多卡 RTX 3090 | Qwen-VL | Whisper-v3"
                        "</span></div>",
                        elem_classes="footer-text",
                    )
    btn_process.click(process_upload, [video_in], [status_display, gallery])
    def user_msg(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]
    def bot_msg(history):
        if len(history) > 0:
            raw_query = history[-1]["content"]
            query = raw_query
            if isinstance(raw_query, list):
                for item in raw_query:
                    if isinstance(item, dict) and item.get("type") == "text":
                        query = item["text"]
                        break
        else:
            query = ""
        response, images = chat_engine(query, history)
        history.append({"role": "assistant", "content": response})
        return history, images
    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_msg, [chatbot], [chatbot, gallery]
    )
    btn_send.click(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_msg, [chatbot], [chatbot, gallery]
    )
    btn_clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True,
        theme=theme,
        css=custom_css
    )