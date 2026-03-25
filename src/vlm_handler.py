import os
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from absorption_layer import AbsorptionLayer, build_absorption_config
    from visual_feature_extractor import VisualFeatureExtractor
except ImportError:
    from src.absorption_layer import AbsorptionLayer, build_absorption_config
    from src.visual_feature_extractor import VisualFeatureExtractor


class VLMHandler:
    def __init__(self, max_retries=3, retry_delay_seconds=8):
        print("[VLM] Loading Qwen2.5-VL...")

        self.processor = None
        self.model = None
        self.available = False
        self.load_error = None

        default_device = (
            "cuda:1"
            if torch.cuda.device_count() > 1
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.device = os.getenv("VLM_DEVICE", default_device)
        try:
            self.max_visual_images = int(os.getenv("VLM_MAX_VISUAL_IMAGES", "6"))
        except ValueError:
            self.max_visual_images = 6
        try:
            self.max_mcq_images = int(os.getenv("VLM_MAX_MCQ_IMAGES", "12"))
        except ValueError:
            self.max_mcq_images = 12
        self.use_logprob_mcq = os.getenv("VLM_MCQ_USE_LOGPROB", "1") != "0"

        # Absorption Layer 配置
        self.use_absorption = os.getenv("VLM_USE_ABSORPTION", "0") == "1"
        self.absorption = None
        self.visual_extractor = None

        local_path = os.getenv("VLM_LOCAL_PATH", "./Qwen2.5-VL-7B-Instruct")
        remote_model_id = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
        model_path = local_path if os.path.exists(local_path) else remote_model_id

        if Qwen2_5_VLForConditionalGeneration is None:
            self.load_error = (
                "Current transformers does not include Qwen2.5-VL class. "
                "Please upgrade to transformers>=4.49.0 (recommended: latest stable)."
            )
            print(f"[Error] {self.load_error}")
            return

        model_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[VLM] Loading attempt {attempt}/{max_retries} from {model_path}")
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                )
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype=model_dtype,
                ).eval()
                self.model.to(self.device)
                self.available = True
                self.load_error = None
                print(f"[VLM] Model loaded successfully! (Running on {self.device})")

                # 初始化 Absorption Layer（如果启用）
                if self.use_absorption:
                    self._init_absorption_layer()
                break
            except Exception as e:
                self.load_error = str(e)
                print(f"[Error] Model loading failed on attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    wait_seconds = retry_delay_seconds * attempt
                    print(f"[VLM] Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)

        if not self.available:
            print("[Error] Qwen2.5-VL is unavailable after retries.")
            print("Please check network/HF mirror and transformers version (recommended: >=4.49.0).")

    @staticmethod
    def _detect_query_language(query: str) -> str:
        """Detect the dominant language in the user query."""
        if not query:
            return "unknown"

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", query))
        latin_count = len(re.findall(r"[A-Za-z]", query))

        if cjk_count > 0 and cjk_count >= latin_count:
            return "zh"
        if latin_count > 0:
            return "en"
        return "unknown"

    def _build_language_instruction(self, query: str) -> str:
        """Build a strong output-language constraint for the prompt."""
        detected_lang = self._detect_query_language(query)

        if detected_lang == "zh":
            return (
                "Output Language: Simplified Chinese.\n"
                "You must answer in Simplified Chinese unless the user explicitly asks for another language.\n"
            )
        if detected_lang == "en":
            return (
                "Output Language: English.\n"
                "You must answer in English unless the user explicitly asks for another language.\n"
            )
        return (
            "Output Language: Match the user's query language.\n"
            "Do not switch languages unless the user explicitly asks for another language.\n"
        )

    def _init_absorption_layer(self):
        """初始化 AbsorptionLayer 和 VisualFeatureExtractor。"""
        try:
            cfg = build_absorption_config()
            model_dtype = next(self.model.parameters()).dtype
            self.absorption = AbsorptionLayer(**cfg).to(
                device=self.device, dtype=model_dtype
            ).eval()

            # 尝试加载预训练权重
            weight_path = os.getenv("ABSORPTION_WEIGHT_PATH", "")
            if weight_path and os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location=self.device)
                self.absorption.load_state_dict(state_dict)
                print(f"[Absorption] Loaded weights from {weight_path}")
            else:
                param_count = sum(p.numel() for p in self.absorption.parameters())
                print(f"[Absorption] Initialized with random weights ({param_count/1e6:.1f}M params)")
                print("[Absorption] Warning: Using untrained weights. Run train_absorption.py first for best results.")

            # 设置 VisualFeatureExtractor
            method = os.getenv("ABSORPTION_VISUAL_METHOD", "auto")
            self.visual_extractor = VisualFeatureExtractor(
                method=method,
                d_model=cfg["d_model"],
            )
            # 尝试注册 Qwen-VL ViT hook
            hook_ok = self.visual_extractor.setup_qwen_vl_hook(self.model)
            if hook_ok:
                print("[Absorption] Qwen-VL ViT hook registered successfully")
            else:
                print("[Absorption] Qwen-VL hook failed, will fallback to CLIP if available")

            print("[Absorption] Layer initialized and ready")
        except Exception as e:
            print(f"[Absorption] Initialization failed: {e}. Falling back to native mode.")
            self.use_absorption = False
            self.absorption = None

    def _prepare_absorption_inputs(
        self,
        prompt_text: str,
        pil_images: list,
        timestamps_sec: list = None,
    ) -> dict:
        """Build model inputs for absorption path (inputs_embeds + attention_mask)."""
        if not pil_images:
            raise ValueError("Absorption path requires at least one image")

        # Step 1: 提取视觉特征
        visual_features, _ = self.visual_extractor.extract(
            images=pil_images,
            qwen_model=self.model,
            qwen_processor=self.processor,
            device_qwen=self.device,
        )

        # Step 2: 纯文本 tokenize（不传图片，避免生成视觉占位符 token）
        text_only_messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text_str = self.processor.apply_chat_template(
            text_only_messages, tokenize=False, add_generation_prompt=True,
        )
        text_inputs = self.processor.tokenizer(
            text_str, return_tensors="pt", padding=True,
        ).to(self.device)
        input_ids = text_inputs["input_ids"]  # (1, N)
        attention_mask = text_inputs.get("attention_mask")

        # Step 3: 获取文本 embedding
        with torch.no_grad():
            text_embeds = self.model.get_input_embeddings()(input_ids)  # (1, N, d_model)

        # 构建时间戳
        timestamps_tensor = None
        if timestamps_sec is not None:
            M = visual_features.shape[1]
            tokens_per_frame = M // max(len(timestamps_sec), 1)
            tokens_per_frame = max(tokens_per_frame, 1)
            timestamps_tensor = VisualFeatureExtractor.build_timestamps_for_frames(
                num_frames=len(timestamps_sec),
                timestamps_sec=timestamps_sec,
                tokens_per_frame=tokens_per_frame,
            ).to(self.device)
            # 截断/填充到实际 M
            if timestamps_tensor.shape[1] > M:
                timestamps_tensor = timestamps_tensor[:, :M]
            elif timestamps_tensor.shape[1] < M:
                pad = timestamps_tensor[:, -1:].expand(-1, M - timestamps_tensor.shape[1])
                timestamps_tensor = torch.cat([timestamps_tensor, pad], dim=1)

        # Step 4: Absorption — 视觉信息被压缩进文本 token
        visual_features = visual_features.to(dtype=text_embeds.dtype, device=self.device)
        with torch.no_grad():
            t_fused = self.absorption(
                text_embeds=text_embeds,
                visual_embeds=visual_features,
                timestamps=timestamps_tensor,
            )  # (1, N, d_model)

        model_inputs = {
            "inputs_embeds": t_fused,
            "attention_mask": attention_mask,
        }
        return model_inputs

    def _inference_with_absorption(
        self,
        prompt_text: str,
        pil_images: list,
        timestamps_sec: list = None,
    ) -> str:
        """通过吞噬层进行推理：视觉 token 被压缩进文本 token。"""
        if not pil_images:
            # 无图片时退回纯文本推理
            return self._text_only_generate(prompt_text)

        model_inputs = self._prepare_absorption_inputs(
            prompt_text=prompt_text,
            pil_images=pil_images,
            timestamps_sec=timestamps_sec,
        )

        generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "inputs_embeds": model_inputs["inputs_embeds"],
        }
        if model_inputs.get("attention_mask") is not None:
            generate_kwargs["attention_mask"] = model_inputs["attention_mask"]

        generated_ids = self.model.generate(**generate_kwargs)
        prompt_len = model_inputs["inputs_embeds"].shape[1]
        generated_ids_trimmed = generated_ids[:, prompt_len:]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""

    def _text_only_generate(self, prompt_text: str) -> str:
        """纯文本推理（无图片时的回退路径）。"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor.tokenizer(
            text_str, return_tensors="pt", padding=True,
        ).to(self.device)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
        )
        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        return output[0] if output else ""

    def chat(self, query, images_info, audio_info):
        """Generate answer with multimodal context"""
        if not self.available or self.model is None or self.processor is None:
            error_msg = self.load_error or "Unknown model initialization error."
            return (
                "[Model Unavailable] Qwen2.5-VL failed to initialize. "
                f"Reason: {error_msg}. "
                "Please restart later, or set VLM_LOCAL_PATH/VLM_MODEL_ID explicitly."
            )

        pil_images = []
        visual_context = "Visual Evidence (Screenshots):\n"
        for i, (ts, score, path) in enumerate(images_info[: self.max_visual_images]):
            m, s = divmod(int(ts), 60)
            if not path or not os.path.exists(path):
                visual_context += f"Image {i+1}: Timestamp {m:02d}:{s:02d} (missing image file)\n"
                continue

            try:
                with Image.open(path) as img:
                    pil_images.append(img.convert("RGB"))
                visual_context += f"Image {i+1}: Timestamp {m:02d}:{s:02d}\n"
            except Exception as e:
                visual_context += f"Image {i+1}: Timestamp {m:02d}:{s:02d} (load failed: {e})\n"

        audio_context = "\nAudio Transcript Evidence (Teacher's speech):\n"
        if not audio_info:
            audio_context += "(No relevant audio found)\n"
        else:
            for i, (ts, text, score) in enumerate(audio_info):
                if i >= 10:
                    break
                m, s = divmod(int(ts), 60)
                audio_context += f"- At {m:02d}:{s:02d}: \"{text}\"\n"

        language_instruction = self._build_language_instruction(query)
        prompt_instruction = (
            f"{language_instruction}\n"
            f"{visual_context}"
            f"{audio_context}\n"
            f"User Query: {query}\n\n"
            "Instructions:\n"
            "1. **Synthesize**: Combine the visual slides (OCR) and the teacher's speech to answer.\n"
            "2. **List Extraction**: If the user asks for a list (e.g., universities), extract unique names from the slides/audio. **Do not repeat names.**\n"
            "3. **Priority**: If the visual text is blurry, RELY on the Audio Transcript.\n"
            "4. **Concise**: Give a direct and summarized answer.\n"
            "5. **Language**: Follow the required output language above. "
            "Do not switch to another language unless the user explicitly requests it."
        )

        print("[VLM] Fusion prompt constructed. Sending to model...")

        # === Absorption Layer 路径 ===
        if self.use_absorption and self.absorption is not None and pil_images:
            try:
                timestamps_sec = [ts for ts, _, _ in images_info[:self.max_visual_images]]
                result = self._inference_with_absorption(
                    prompt_text=prompt_instruction,
                    pil_images=pil_images,
                    timestamps_sec=timestamps_sec,
                )
                print("[VLM] Absorption inference completed")
                return result
            except Exception as e:
                print(f"[VLM] Absorption inference failed, falling back to native: {e}")

        # === 原生路径（向后兼容） ===
        message_content = [{"type": "image"} for _ in pil_images]
        message_content.append({"type": "text", "text": prompt_instruction})
        messages = [{"role": "user", "content": message_content}]

        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            processor_kwargs = {
                "text": [text],
                "padding": True,
                "return_tensors": "pt",
            }
            if pil_images:
                processor_kwargs["images"] = pil_images

            inputs = self.processor(**processor_kwargs).to(self.device)
            generated_ids = self.model.generate(
                **inputs,
                repetition_penalty=1.2,
                max_new_tokens=512,
                do_sample=False,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text[0] if output_text else ""
        except Exception as e:
            return f"[Model Error] {str(e)}"

    def answer_mcq(
        self,
        question: str,
        options: list,
        image_paths: list = None,
        subtitles: str = None,
        with_subtitles: bool = True
    ) -> tuple:
        """
        Answer a multiple-choice question for Video-MME benchmark.
        
        Uses the official Video-MME prompt format:
        "Select the best answer to the following multiple-choice question 
        based on the video. Respond with only the letter (A, B, C, or D) 
        of the correct option."
        
        Args:
            question: The question text
            options: List of options ["A. xxx", "B. xxx", "C. xxx", "D. xxx"]
            image_paths: List of paths to frame images
            subtitles: Subtitle text (optional, for with-subtitles evaluation)
            with_subtitles: Whether to include subtitles in prompt
            
        Returns:
            Tuple of (extracted_answer, raw_response)
            - extracted_answer: Single letter A, B, C, or D (or None if extraction failed)
            - raw_response: Full model response
        """
        if not self.available or self.model is None:
            error_msg = self.load_error or "Model not loaded"
            return None, f"[Model Unavailable] {error_msg}"

        prompt_text = self._build_mcq_prompt(
            question=question,
            options=options,
            subtitles=subtitles,
            with_subtitles=with_subtitles,
        )
        pil_images = self._load_pil_images(
            image_paths=image_paths,
            max_images=self.max_mcq_images,
        )

        try:
            # === Absorption Layer MCQ 路径 ===
            if self.use_absorption and self.absorption is not None and pil_images:
                try:
                    absorption_inputs = self._prepare_absorption_inputs(
                        prompt_text=prompt_text,
                        pil_images=pil_images,
                    )

                    # Primary path in absorption mode: direct option scoring by logprob.
                    if self.use_logprob_mcq:
                        score_result = self._score_mcq_by_logprob(absorption_inputs)
                        if score_result is not None:
                            answer, score_text = score_result
                            return answer, f"[Absorption]{score_text}"

                    # Fallback in absorption mode: short generation + regex extraction.
                    answer, raw = self._answer_mcq_by_generation(absorption_inputs)
                    return answer, f"[Absorption] {raw}"
                except Exception as e:
                    print(f"[MCQ] Absorption failed, falling back: {e}")

            # === 原生路径 ===
            inputs = self._build_mcq_inputs(prompt_text=prompt_text, pil_images=pil_images)

            # Primary path: direct option scoring to avoid generation-format drift.
            if self.use_logprob_mcq:
                score_result = self._score_mcq_by_logprob(inputs)
                if score_result is not None:
                    answer, score_text = score_result
                    return answer, score_text

            # Fallback: constrained short generation and regex extraction.
            return self._answer_mcq_by_generation(inputs)
        except Exception as e:
            return None, f"[Model Error] {str(e)}"

    def _build_mcq_prompt(
        self,
        question: str,
        options: list,
        subtitles: str = None,
        with_subtitles: bool = True,
    ) -> str:
        """Build benchmark MCQ prompt text."""
        prompt_parts = []

        if with_subtitles and subtitles:
            prompt_parts.append("This video's subtitles are listed below:")
            prompt_parts.append(subtitles)
            prompt_parts.append("")

        prompt_parts.append(
            "Select the best answer to the following multiple-choice question "
            "based on the video. Respond with only the letter (A, B, C, or D) "
            "of the correct option."
        )
        prompt_parts.append("")
        prompt_parts.append(question)
        prompt_parts.append("")

        for opt in options:
            prompt_parts.append(str(opt))

        prompt_parts.append("")
        prompt_parts.append("The best answer is:")
        return "\n".join(prompt_parts)

    def _load_pil_images(self, image_paths: list = None, max_images: int = None) -> list:
        """Load frame paths as RGB PIL images."""
        pil_images = []
        if not image_paths:
            return pil_images

        selected_paths = image_paths
        if max_images is not None and max_images > 0 and len(image_paths) > max_images:
            selected_paths = image_paths[:max_images]
            print(
                f"[MCQ] Truncated image inputs from {len(image_paths)} to {len(selected_paths)} "
                f"(VLM_MAX_MCQ_IMAGES={max_images})"
            )

        for path in selected_paths:
            try:
                with Image.open(path) as img:
                    pil_images.append(img.convert("RGB"))
            except Exception as e:
                print(f"[Warning] Failed to load image {path}: {e}")
        return pil_images

    def _build_mcq_inputs(self, prompt_text: str, pil_images: list):
        """Build model-ready inputs for MCQ inference."""
        message_content = [{"type": "image"} for _ in pil_images]
        message_content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": message_content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if pil_images:
            processor_kwargs["images"] = pil_images

        return self.processor(**processor_kwargs).to(self.device)

    def _score_mcq_by_logprob(self, inputs) -> tuple:
        """
        Score A/B/C/D directly using conditional log probability.

        Returns:
            Tuple(answer_letter, score_text) or None if scoring failed.
        """
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)[0]

            scores = {}
            tokenizer = self.processor.tokenizer
            for letter in ("A", "B", "C", "D"):
                scores[letter] = self._score_letter_candidate(
                    letter=letter,
                    tokenizer=tokenizer,
                    next_token_logprobs=next_token_logprobs,
                    base_inputs=inputs,
                )

            valid_scores = {k: v for k, v in scores.items() if v != float("-inf")}
            if not valid_scores:
                return None

            answer = max(valid_scores.items(), key=lambda x: x[1])[0]
            score_text = (
                "[LogProb] "
                + " ".join([f"{k}={scores[k]:.4f}" for k in ("A", "B", "C", "D")])
                + f" -> {answer}"
            )
            return answer, score_text
        except Exception as e:
            print(f"[Warning] Logprob scoring failed, fallback to generation: {e}")
            return None

    def _score_letter_candidate(self, letter, tokenizer, next_token_logprobs, base_inputs):
        """
        Score one answer letter.

        Prefer single-token variants (fast); fallback to sequence scoring when needed.
        """
        single_token_scores = []
        seq_variant_ids = []
        variants = [f" {letter}", letter, f"\n{letter}", f"({letter})", f"{letter}."]
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue
            if len(token_ids) == 1:
                single_token_scores.append(next_token_logprobs[token_ids[0]].item())
            else:
                seq_variant_ids.append(token_ids)

        if single_token_scores:
            return max(single_token_scores)

        seq_scores = [self._score_token_sequence(base_inputs, token_ids) for token_ids in seq_variant_ids]
        if seq_scores:
            return max(seq_scores)

        return float("-inf")

    def _score_token_sequence(self, base_inputs, token_ids):
        """Autoregressive sequence logprob for a candidate token sequence."""
        if not token_ids:
            return float("-inf")

        model_inputs = {}
        for k, v in base_inputs.items():
            if torch.is_tensor(v):
                model_inputs[k] = v

        total = 0.0
        with torch.no_grad():
            for token_id in token_ids:
                outputs = self.model(**model_inputs)
                step_logprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
                total += step_logprobs[0, token_id].item()

                if "input_ids" in model_inputs:
                    next_token = torch.tensor(
                        [[token_id]],
                        dtype=model_inputs["input_ids"].dtype,
                        device=model_inputs["input_ids"].device,
                    )
                    model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], next_token], dim=1)
                elif "inputs_embeds" in model_inputs:
                    next_token_ids = torch.tensor(
                        [[token_id]],
                        dtype=torch.long,
                        device=model_inputs["inputs_embeds"].device,
                    )
                    next_embed = self.model.get_input_embeddings()(next_token_ids)
                    next_embed = next_embed.to(dtype=model_inputs["inputs_embeds"].dtype)
                    model_inputs["inputs_embeds"] = torch.cat(
                        [model_inputs["inputs_embeds"], next_embed], dim=1
                    )
                else:
                    return float("-inf")

                if "attention_mask" in model_inputs and model_inputs["attention_mask"] is not None:
                    one = torch.ones(
                        (model_inputs["attention_mask"].shape[0], 1),
                        dtype=model_inputs["attention_mask"].dtype,
                        device=model_inputs["attention_mask"].device,
                    )
                    model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], one], dim=1)
        return total

    def _answer_mcq_by_generation(self, inputs) -> tuple:
        """Fallback path when logprob scoring is unavailable."""
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

        if "input_ids" in inputs and inputs["input_ids"] is not None:
            prompt_len = inputs["input_ids"].shape[1]
        elif "inputs_embeds" in inputs and inputs["inputs_embeds"] is not None:
            prompt_len = inputs["inputs_embeds"].shape[1]
        else:
            prompt_len = 0

        generated_ids_trimmed = generated_ids[:, prompt_len:]

        raw_response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0] if generated_ids_trimmed.numel() > 0 else ""

        extracted = self._extract_mcq_answer(raw_response)
        return extracted, raw_response
    
    def _extract_mcq_answer(self, response: str) -> str:
        """
        Extract single letter answer (A, B, C, or D) from model response.
        
        Handles various response formats:
        - "A"
        - "A."
        - "A. Apples"
        - "The answer is A"
        - "A is correct"
        """
        import re
        
        if not response:
            return None
        
        response = response.strip()
        
        # Pattern 1: Starts with single letter
        match = re.match(r'^([A-D])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: "The answer is X" or "answer: X"
        match = re.search(r'answer\s*(?:is|:)\s*([A-D])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: Any standalone A, B, C, or D
        match = re.search(r'\b([A-D])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 4: Letter followed by period or colon
        match = re.search(r'([A-D])[.:]', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None
