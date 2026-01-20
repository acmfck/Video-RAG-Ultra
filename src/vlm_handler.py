import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class VLMHandler:
    def __init__(self):
        print("[VLM] Loading Qwen-VL-Chat...")
        
        local_path = "./Qwen-VL-Chat"
        model_path = local_path if os.path.exists(local_path) else "Qwen/Qwen-VL-Chat"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="cuda:1", 
                trust_remote_code=True, 
                bf16=True 
            ).eval()
            
            self.model.generation_config.repetition_penalty = 1.2
            
            print(f"[VLM] Model loaded successfully! (Running on cuda:1)")
        except Exception as e:
            print(f"[Error] Model loading failed: {e}")
            print("Please check if transformers version is 4.37.2")

    def chat(self, query, images_info, audio_info):
        """Generate answer with multimodal context"""
        qwen_input_list = []
        
        visual_context = "Visual Evidence (Screenshots):\n"
        for i, (ts, score, path) in enumerate(images_info):
            qwen_input_list.append({'image': path})
            m, s = divmod(int(ts), 60)
            visual_context += f"Image {i+1}: Timestamp {m:02d}:{s:02d}\n"
        
        audio_context = "\nAudio Transcript Evidence (Teacher's speech):\n"
        if not audio_info:
            audio_context += "(No relevant audio found)\n"
        else:
            for i, (ts, text, score) in enumerate(audio_info):
                if i >= 10:
                    break
                m, s = divmod(int(ts), 60)
                audio_context += f"- At {m:02d}:{s:02d}: \"{text}\"\n"

        prompt_instruction = (
            f"{visual_context}"
            f"{audio_context}\n"
            f"User Query: {query}\n\n"
            "Instructions:\n"
            "1. **Synthesize**: Combine the visual slides (OCR) and the teacher's speech to answer.\n"
            "2. **List Extraction**: If the user asks for a list (e.g., universities), extract unique names from the slides/audio. **Do not repeat names.**\n"
            "3. **Priority**: If the visual text is blurry, RELY on the Audio Transcript.\n"
            "4. **Concise**: Give a direct and summarized answer."
        )
        
        qwen_input_list.append({'text': prompt_instruction})
        
        print(f"[VLM] Fusion prompt constructed. Sending to model...")

        try:
            query_formatted = self.tokenizer.from_list_format(qwen_input_list)
            
            response, history = self.model.chat(
                self.tokenizer, 
                query=query_formatted, 
                history=None,
                repetition_penalty=1.2,
                temperature=0.3,
                top_p=0.8,
                max_new_tokens=512
            )
            return response
        except Exception as e:
            return f"[Model Error] {str(e)}"