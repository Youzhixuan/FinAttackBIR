# =============================================================================
# PAIR Framework - Moral Dilemma Attack Integration
# Version: 2.1
# Date: 2025-10-22
# 
# Core Modifications in v2.1:
# 1. Fixed Target response generation bug (line 77: 0 > 0 → TARGET_TEMP > 0)
# 2. Added debug output for Target response generation
# 3. Improved error handling in get_response method
#
# Changes from v2.0:
# - Line 77-79: Fixed temperature condition logic
# - Line 83-85: Added debug output for response generation
#
# =============================================================================
# === START: New and fixed conversers.py ===
from common import get_api_key, conv_template, extract_json
from language_models import APILiteLLM
from config import FASTCHAT_TEMPLATE_NAMES, Model, HF_MODEL_NAMES, TARGET_TEMP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 本地模型加载类 (使用transformers官方范式)
# Modified: 2025-10-25 - CPU Offload: 模型初始加载到CPU，使用时临时移到GPU
# Modified: 2025-10-24 - 添加experiment_logger支持
# Modified: 2025-10-22 - 移除量化，使用fp16以提高精度
class LocalModel:
    def __init__(self, model_name_str: str, experiment_logger=None):
        self.experiment_logger = experiment_logger
        model_path = HF_MODEL_NAMES[Model(model_name_str)]
        print(f"[INFO] Loading local model from path: {model_path}")
        print("[INFO] Using fp16 precision with CPU offload strategy...")
        print("[INFO] Model will stay on CPU and move to GPU only when needed")
        # 使用transformers标准范式加载模型
        # CPU Offload: 初始加载到CPU，生成时临时移到GPU（节省显存）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu"  # 加载到CPU而不是GPU
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 设置pad_token避免警告
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 为Llama-3.1设置正确的chat template
        if "llama-3.1" in model_path.lower() or "llama-3-1" in model_path.lower():
            print("[INFO] Setting Llama-3.1 chat template...")
            self.tokenizer.chat_template = (
                "{% if messages[0]['role'] == 'system' %}"
                    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' + messages[0]['content'] + '<|eot_id|>' }}"
                    "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                    "{{ '<|begin_of_text|>' }}"
                    "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{% endif %}"
            )
        # 为Llama-2设置正确的chat template
        elif self.tokenizer.chat_template is None:
            print("[INFO] Setting Llama-2 chat template...")
            # Llama-2官方chat template格式
            self.tokenizer.chat_template = (
                "<s>"
                "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                        "[INST] <<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n"
                    "{% elif message['role'] == 'user' %}"
                        "{{ message['content'] }} [/INST]"
                    "{% elif message['role'] == 'assistant' %}"
                        " {{ message['content'] }} </s>"
                    "{% endif %}"
                    "{% if not loop.last and messages[loop.index0 + 1]['role'] == 'user' and message['role'] == 'assistant' %}"
                        "<s>"
                    "{% endif %}"
                "{% endfor %}"
            )

        self.template_name = FASTCHAT_TEMPLATE_NAMES[Model(model_name_str)]
        self.gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Local model loaded to CPU successfully.")
        print(f"[INFO] Will use {self.gpu_device} for inference when needed")
        # 这个属性用于AttackLM的提示词格式化
        self.use_open_source_model = True
        # 根据模型类型设置post_message（不同模型有不同的后缀token）
        if "llama-3.1" in model_path.lower() or "llama-3-1" in model_path.lower():
            self.post_message = ""  # Llama-3.1使用空后缀
        else:
            self.post_message = "</s>"  # Llama-2的后缀标记

    # Modified: 2025-12-11 - Added logits_processor parameter for symbol vocabulary control
    # Modified: 2025-12-18 - Added min_new_tokens parameter to prevent early EOS
    # Modified: 2025-12-30 - Added try-finally to ensure model moves back to CPU even on OOM
    def batched_generate(self, convs_list: list[list[dict]], max_n_tokens: int, temperature: float, top_p: float, extra_eos_tokens: list[str] = None, logits_processor=None, min_new_tokens: int = None) -> list[str]:
        # 1025: CPU Offload - Attack生成前，将模型移到GPU
        print(f"[INFO] Moving Attack/Target model to GPU for generation...")
        import time
        start_time = time.time()
        self.model = self.model.to(self.gpu_device)
        move_time = time.time() - start_time
        print(f"[INFO] Model moved to GPU in {move_time:.2f}s")
        
        try:
            # Modified: 2025-11-22 - 移除JSON prefill，直接输出后缀文本
            prompt_texts = [self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in convs_list]
            
            inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.gpu_device)

            gen_kwargs = {"max_new_tokens": max_n_tokens}
            if temperature > 0:
                gen_kwargs['do_sample'] = True
                gen_kwargs['temperature'] = temperature
                gen_kwargs['top_p'] = top_p
                gen_kwargs['repetition_penalty'] = 1.1  # Avoid repeating same suffix
            
            # Modified: 2025-12-18 - Add min_new_tokens to prevent early EOS
            if min_new_tokens is not None:
                gen_kwargs['min_new_tokens'] = min_new_tokens
            
            # Modified: 2025-12-11 - Add logits_processor for symbol vocabulary control
            if logits_processor is not None:
                gen_kwargs['logits_processor'] = logits_processor
            
            output_ids = self.model.generate(**inputs, **gen_kwargs)
            input_lengths = inputs['input_ids'].shape[1]
            raw_responses = self.tokenizer.batch_decode(output_ids[:, input_lengths:], skip_special_tokens=True)
            
            # 直接使用原始输出
            responses = raw_responses
            return [resp.strip() for resp in responses]
        
        finally:
            # 2025-12-30: 确保无论成功还是OOM，模型都移回CPU
            print(f"[INFO] Moving Attack/Target model back to CPU...")
            start_time = time.time()
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            move_time = time.time() - start_time
            print(f"[INFO] Model moved to CPU in {move_time:.2f}s, GPU freed")

    def get_response(self, prompts_list: list[str]) -> list[str]:
        # v2.1: 修复了temperature逻辑并添加debug输出
        # v2.2: 移除debug print，改用experiment_logger
        # v2.3: 使用config常量避免硬编码
        # v2.4: CPU Offload - 生成时临时移到GPU
        # v2.5: 2025-12-30 - Added try-finally to ensure model moves back to CPU even on OOM
        from config import TARGET_MAX_N_TOKENS
        
        # 1025: CPU Offload - Target生成前，将模型移到GPU
        print(f"[INFO] Moving Attack/Target model to GPU for generation...")
        import time
        start_time = time.time()
        self.model = self.model.to(self.gpu_device)
        move_time = time.time() - start_time
        print(f"[INFO] Model moved to GPU in {move_time:.2f}s")
        
        try:
            responses = []
            full_prompts = []
            raw_outputs = []
            
            for idx, prompt in enumerate(prompts_list):
                conv = conv_template(self.template_name)
                conv.append_message(conv.roles[0], prompt)
                full_prompt = conv.get_prompt()
                # 1025: 确保Llama-2的prompt以[/INST]结尾
                if "llama-2" in self.template_name and not full_prompt.strip().endswith("[/INST]"):
                    full_prompt += " [/INST]"
                full_prompts.append(full_prompt)

                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.gpu_device)

                # v2.1修复: 使用TARGET_TEMP而不是硬编码的0
                # v2.3: 使用TARGET_MAX_N_TOKENS常量而不是硬编码
                gen_kwargs = {"max_new_tokens": TARGET_MAX_N_TOKENS}
                if TARGET_TEMP > 0: # v2.1: 修复了条件判断 (原来是 0 > 0)
                    gen_kwargs['do_sample'] = True
                    gen_kwargs['temperature'] = TARGET_TEMP

                output_ids = self.model.generate(**inputs, **gen_kwargs)
                input_length = inputs['input_ids'].shape[1]
                
                # 保存原始输出（包括特殊token）用于logger
                raw_response_with_special = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=False)
                raw_outputs.append(raw_response_with_special)
                
                response = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
                responses.append(response)
            
            # 返回responses和额外信息供TargetLM记录
            return responses, full_prompts, raw_outputs
        
        finally:
            # 2025-12-30: 确保无论成功还是OOM，模型都移回CPU
            print(f"[INFO] Moving Attack/Target model back to CPU...")
            start_time = time.time()
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            move_time = time.time() - start_time
            print(f"[INFO] Model moved to CPU in {move_time:.2f}s, GPU freed")

def load_attack_and_target_models(args, experiment_logger=None):
    # Modified: 2025-10-24 - 添加experiment_logger支持
    # Modified: 2025-10-22 - 添加模型共享逻辑以节省显存
    if args.evaluate_locally:
        # 检查是否为自我攻击模式（attack和target使用同一模型）
        if args.attack_model == args.target_model:
            print("[INFO] Using shared model mode (self-attack)")
            print(f"[INFO] Loading shared model: {args.attack_model}")
            print("[INFO] This will save GPU memory by loading the model only once")
            
            # 只加载一次模型
            shared_model = load_indiv_model(args.attack_model, local=True, use_jailbreakbench=False, experiment_logger=experiment_logger)
            
            # 将共享模型注入到AttackLM和TargetLM
            attackLM = AttackLM(
                model_name=args.attack_model, 
                max_n_tokens=args.attack_max_n_tokens, 
                max_n_attack_attempts=args.max_n_attack_attempts, 
                category=args.category, 
                evaluate_locally=True,
                shared_model=shared_model,  # 注入共享模型
                experiment_logger=experiment_logger
            )
            targetLM = TargetLM(
                model_name=args.target_model, 
                category=args.category, 
                max_n_tokens=args.target_max_n_tokens, 
                evaluate_locally=True, 
                phase=args.jailbreakbench_phase,
                shared_model=shared_model,  # 注入同一个共享模型
                experiment_logger=experiment_logger
            )
        else:
            # 不同模型的情况：分别加载
            print("[INFO] Using separate model mode (different attack and target models)")
            attackLM = AttackLM(
                model_name=args.attack_model, 
                max_n_tokens=args.attack_max_n_tokens, 
                max_n_attack_attempts=args.max_n_attack_attempts, 
                category=args.category, 
                evaluate_locally=True,
                experiment_logger=experiment_logger
            )
            targetLM = TargetLM(
                model_name=args.target_model, 
                category=args.category, 
                max_n_tokens=args.target_max_n_tokens, 
                evaluate_locally=True, 
                phase=args.jailbreakbench_phase,
                experiment_logger=experiment_logger
            )
    else:
        print("[INFO] Using API evaluation mode.")
        attackLM = AttackLM(
            model_name=args.attack_model, 
            max_n_tokens=args.attack_max_n_tokens, 
            max_n_attack_attempts=args.max_n_attack_attempts, 
            category=args.category, 
            evaluate_locally=False,
            experiment_logger=experiment_logger
        )
        targetLM = TargetLM(
            model_name=args.target_model, 
            category=args.category, 
            max_n_tokens=args.target_max_n_tokens, 
            evaluate_locally=False, 
            phase=args.jailbreakbench_phase,
            experiment_logger=experiment_logger
        )
    return attackLM, targetLM

def load_indiv_model(model_name, local=False, use_jailbreakbench=True, experiment_logger=None):
    if local:
        # This is the path we now use for local models
        return LocalModel(model_name, experiment_logger=experiment_logger)

    # Original API logic remains untouched
    if use_jailbreakbench:
        from jailbreakbench import LLMLiteLLM
        api_key = get_api_key(Model(model_name))
        return LLMLiteLLM(model_name=model_name, api_key=api_key)
    else:
        return APILiteLLM(model_name)

class AttackLM:
    # Modified: 2025-10-24 - 添加experiment_logger支持
    # Modified: 2025-10-22 - 添加shared_model支持以实现模型共享
    def __init__(self, model_name: str, max_n_tokens: int, max_n_attack_attempts: int, category: str, evaluate_locally: bool, shared_model=None, experiment_logger=None):
        self.model_name = Model(model_name)
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        from config import ATTACK_TEMP, ATTACK_TOP_P
        self.temperature = ATTACK_TEMP
        self.top_p = ATTACK_TOP_P
        self.category = category
        self.evaluate_locally = evaluate_locally
        self.experiment_logger = experiment_logger
        
        # 依赖注入：如果提供了共享模型，使用它；否则自己加载
        if shared_model is not None:
            print("[INFO] AttackLM using shared model (dependency injection)")
            self.model = shared_model
            self.initialize_output = self.model.use_open_source_model
            self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]
        else:
            # 原有逻辑：自己加载模型
            self.model = load_indiv_model(model_name, local=evaluate_locally, use_jailbreakbench=False, experiment_logger=experiment_logger)
            self.initialize_output = self.model.use_open_source_model
            self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

    def preprocess_conversation(self, convs_list: list, prompts_list: list[str]):
        init_message = ""
        if self.initialize_output:
            # v2.3: 预填充简化的JSON开头（只有prompt字段）
            init_message = '{"prompt": "'
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)
        openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
        return openai_convs_list, init_message

    def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
        # v2.2: 添加experiment_logger支持
        # v2.1.1: 移除extra_eos_tokens以防止JSON被提前截断
        
        # 记录stage开始（只在有logger时记录一次）
        if self.experiment_logger:
            self.experiment_logger.log_stage_start("Attack Prompt Generation")
        
        outputs_list = self.model.batched_generate(
            openai_conv_list,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_eos_tokens=[]  # v2.1.1: 从["}"]改为[]
        )

        valid_outputs = []
        new_adv_prompts = []
        for idx, full_output in enumerate(outputs_list):
            # v2.3: 使用鲁棒提取处理各种JSON格式
            # 构造完整JSON（添加预填充和结尾）
            full_json = init_message + full_output + '"}'
            
            # 使用鲁棒提取函数
            from common import extract_prompt_robust
            extracted_prompt = extract_prompt_robust(full_json)
            
            # 构造字典（保持接口兼容性）
            attack_dict = {
                "improvement": "",  # v2.3: 不需要improvement
                "prompt": extracted_prompt
            }
            
            # 检查prompt是否嵌入了goal（用于processed output）
            has_bomb = 'bomb' in extracted_prompt.lower()
            has_virus = 'virus' in extracted_prompt.lower()
            
            # 记录到experiment logger（只记录第一个stream的详细信息）
            if self.experiment_logger and idx == 0:
                # 从openai_conv_list构建完整的input prompt字符串
                input_prompt_str = str(openai_conv_list[0])
                
                parameters = {
                    'max_new_tokens': self.max_n_tokens,
                    'do_sample': self.temperature > 0,
                    'temperature': self.temperature if self.temperature > 0 else None,
                    'top_p': self.top_p if self.temperature > 0 else None
                }
                
                processed_info = f"Extracted prompt (length={len(extracted_prompt)})\nGoal embedding: bomb={has_bomb}, virus={has_virus}"
                
                self.experiment_logger.log_model_call(
                    model_name=f"Attacker ({self.model_name.value})",
                    parameters=parameters,
                    input_prompt=input_prompt_str,
                    raw_output=full_output,
                    processed_output=processed_info
                )
            
            valid_outputs.append(attack_dict)
            new_adv_prompts.append(extracted_prompt)
        return valid_outputs, new_adv_prompts

    def get_attack(self, convs_list, prompts_list):
        processed_convs_list, init_message = self.preprocess_conversation(convs_list, prompts_list)
        valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)
        for jailbreak_prompt, conv in zip(new_adv_prompts, convs_list):
            if self.initialize_output:
                jailbreak_prompt += self.model.post_message
            conv.update_last_message(jailbreak_prompt)
        return valid_outputs

class TargetLM:
    # Modified: 2025-10-24 - 添加experiment_logger支持
    # Modified: 2025-10-22 - 添加shared_model支持以实现模型共享
    def __init__(self, model_name: str, category: str, max_n_tokens: int, phase: str, evaluate_locally: bool = False, use_jailbreakbench: bool = True, shared_model=None, experiment_logger=None):
        self.model_name = model_name
        self.max_n_tokens = max_n_tokens
        self.phase = phase
        self.evaluate_locally = evaluate_locally
        self.category = category
        self.experiment_logger = experiment_logger
        
        # 依赖注入：如果提供了共享模型，使用它；否则自己加载
        if shared_model is not None:
            print("[INFO] TargetLM using shared model (dependency injection)")
            self.model = shared_model
        else:
            # 原有逻辑：自己加载模型
            self.model = load_indiv_model(model_name, local=evaluate_locally, use_jailbreakbench=not evaluate_locally, experiment_logger=experiment_logger)

    def get_response(self, prompts_list):
        # 记录stage开始
        if self.experiment_logger:
            self.experiment_logger.log_stage_start("Target Model Inference")
        
        # 如果是本地模型，它会返回额外信息
        if self.evaluate_locally:
            responses, full_prompts, raw_outputs = self.model.get_response(prompts_list)
            
            # 记录第一个prompt的详细信息到logger
            if self.experiment_logger and len(responses) > 0:
                from config import TARGET_TEMP, TARGET_MAX_N_TOKENS
                parameters = {
                    'max_new_tokens': TARGET_MAX_N_TOKENS,  # 从config读取，与LocalModel.get_response保持一致
                    'do_sample': TARGET_TEMP > 0,
                    'temperature': TARGET_TEMP if TARGET_TEMP > 0 else None
                }
                
                processed_info = f"Response length: {len(responses[0])} chars"
                
                self.experiment_logger.log_model_call(
                    model_name=f"Target ({self.model_name})",
                    parameters=parameters,
                    input_prompt=full_prompts[0],
                    raw_output=raw_outputs[0],
                    processed_output=processed_info
                )
            
            return responses
        else:
            # API模式，直接返回
            return self.model.get_response(prompts_list)
# ============================================================================
# FinancialTargetLM - FinMA-7B for financial decision attacks
# Created: 2025-11-10
# ============================================================================
class FinancialTargetLM:
    """
    Target model for financial decision attacks.
    Wraps FinMA-7B to generate investment recommendations.
    """
    def __init__(self, model_path="../models/finma-7b-full", experiment_logger=None):
        self.model_path = model_path
        self.experiment_logger = experiment_logger
        self.model_name = "FinMA-7B"
        
        print(f"[INFO] Loading FinMA-7B from {model_path}...")
        print("[INFO] Using fp16 precision with CPU offload strategy...")
        
        # Load model to CPU initially (CPU Offload strategy)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",  # Load to CPU, move to GPU when needed
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] FinMA-7B loaded to CPU successfully")
        print(f"[INFO] Will use {self.gpu_device} for inference when needed")
    
    def generate_single_response(self, full_prompt):
        """
        Generate response for a single prompt.
        MODIFIED: 2025-11-25 to match PIXIU evaluation settings
        Uses greedy decoding (do_sample=False) to ensure deterministic outputs
        consistent with the original evaluation framework.
        
        Args:
            full_prompt: Combined context + adversarial suffix
        
        Returns:
            str: Model response (e.g., "yes", "no", "positive", "negative")
        """
        try:
            # Import config parameters
            from config import TARGET_MAX_N_TOKENS
            
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            # CRITICAL: Use greedy decoding to match PIXIU evaluation - Modified: 2025-11-25
            # Original PIXIU setting: do_sample=False (greedy decoding)
            # This ensures outputs are deterministic and match evaluation behavior
            # Note: TARGET_TEMP is ignored here (greedy decoding doesn't use temperature)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=TARGET_MAX_N_TOKENS,  # Modified: 2025-11-25 - Read from config instead of hardcoded
                    do_sample=False,    # Modified: 2025-11-25 (was True, now greedy decoding)
                    # temperature and top_p removed - not used in greedy decoding
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode full output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part (remove prompt)
            if full_output.startswith(full_prompt):
                response = full_output[len(full_prompt):].strip()
            else:
                response = full_output.strip()
            
            return response
            
        except Exception as e:
            print(f"[ERROR] FinMA-7B generation failed: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def get_response(self, prompts_list):
        """
        Generate responses for a batch of prompts.
        Modified: 2025-12-18 - True batch processing for efficiency
        
        Args:
            prompts_list: List of prompts (each is context + adversarial suffix)
        
        Returns:
            List of responses
        """
        from config import TARGET_MAX_N_TOKENS
        
        if self.experiment_logger:
            self.experiment_logger.log_stage_start("Target Model (FinMA-7B) Generation")
        
        # Move model to GPU for inference (only once!)
        print(f"[INFO] Moving FinMA-7B to GPU for batch generation ({len(prompts_list)} prompts)...")
        import time
        start_time = time.time()
        self.model = self.model.to(self.gpu_device)
        move_time = time.time() - start_time
        print(f"[INFO] Model moved to GPU in {move_time:.2f}s")
        
        # 2025-12-18 - True batch processing
        # 2025-12-30 - Added finally block to ensure model moves back to CPU even on OOM
        try:
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Batch tokenize all prompts (left padding for generation)
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(
                prompts_list, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048  # Prevent OOM from very long prompts
            ).to(self.gpu_device)
            
            # Record input lengths for each prompt (before padding)
            input_lengths = [len(self.tokenizer.encode(p)) for p in prompts_list]
            
            print(f"[INFO] Batch generating {len(prompts_list)} responses...")
            gen_start = time.time()
            
            # Batch generate (greedy decoding for deterministic outputs)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=TARGET_MAX_N_TOKENS,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            gen_time = time.time() - gen_start
            print(f"[INFO] Batch generation completed in {gen_time:.2f}s")
            
            # Decode outputs and extract generated parts
            responses = []
            batch_input_len = inputs['input_ids'].shape[1]  # Padded length
            
            for i, output in enumerate(outputs):
                # Decode the generated part only (after the padded input)
                generated_tokens = output[batch_input_len:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                responses.append(response)
            
            # Log first response if logger available
            if self.experiment_logger and len(responses) > 0:
                self.experiment_logger.log_model_call(
                    model_name="FinMA-7B (Target, Batch)",
                    parameters={
                        'max_new_tokens': TARGET_MAX_N_TOKENS,
                        'do_sample': False,
                        'decoding': 'greedy',
                        'batch_size': len(prompts_list)
                    },
                    input_prompt=prompts_list[0][:500] + "..." if len(prompts_list[0]) > 500 else prompts_list[0],
                    raw_output=responses[0],
                    processed_output=f"Batch size: {len(prompts_list)}, Gen time: {gen_time:.2f}s"
                )
                
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            print(f"[INFO] Falling back to sequential generation...")
            # Fallback to sequential generation
            responses = []
            for idx, prompt in enumerate(prompts_list):
                response = self.generate_single_response(prompt)
                responses.append(response)
        
        finally:
            # 2025-12-30: 确保无论成功还是OOM，模型都移回CPU
            if not getattr(self, '_skip_offload', False):
                print(f"[INFO] Moving FinMA-7B back to CPU...")
                start_time = time.time()
                self.model = self.model.to("cpu")
                torch.cuda.empty_cache()
                move_time = time.time() - start_time
            print(f"[INFO] Model moved to CPU in {move_time:.2f}s, GPU memory freed")
            
            if self.experiment_logger:
                self.experiment_logger.log_stage_end("Target Model (FinMA-7B) Generation")
        
        return responses

# ============================================================================
# XuanYuanTargetLM - XuanYuan-6B for financial decision attacks
# Created: 2026-01-18
# Modified: 2026-02-06 - Removed duplicate definition
# ============================================================================
class XuanYuanTargetLM:
    """
    Target model for financial decision attacks using XuanYuan-6B.
    XuanYuan-6B is a LLaMA-based financial LLM with specific prompt format.
    
    Prompt format: " Human: {content} Assistant:"
    """
    def __init__(self, model_path="../models/XuanYuan-6B", experiment_logger=None):
        self.model_path = model_path
        self.experiment_logger = experiment_logger
        self.model_name = "XuanYuan-6B"
        
        # XuanYuan-6B specific prompt format
        self.seps = [" ", "</s>"]
        self.roles = ["Human", "Assistant"]
        
        print(f"[INFO] Loading XuanYuan-6B from {model_path}...")
        print("[INFO] Using fp16 precision with CPU offload strategy...")
        
        # Load model - XuanYuan-6B uses LlamaForCausalLM but AutoModelForCausalLM works too
        from transformers import LlamaForCausalLM
        
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU, move to GPU when needed
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] XuanYuan-6B loaded to CPU successfully")
        print(f"[INFO] Will use {self.gpu_device} for inference when needed")
    
    def _format_prompt(self, content):
        """
        Format content using XuanYuan-6B's conversation template.
        Format: " Human: {content} Assistant:"
        """
        return self.seps[0] + self.roles[0] + ": " + content + self.seps[0] + self.roles[1] + ":"
    
    def generate_single_response(self, full_prompt):
        """
        Generate response for a single prompt.
        Uses greedy decoding to match PIXIU evaluation settings.
        
        Args:
            full_prompt: Combined context + adversarial suffix (will be wrapped in conversation format)
        
        Returns:
            str: Model response
        """
        try:
            from config import TARGET_MAX_N_TOKENS
            
            # Format prompt using XuanYuan conversation template
            formatted_prompt = self._format_prompt(full_prompt)
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=TARGET_MAX_N_TOKENS,
                    do_sample=False,  # Greedy decoding for deterministic outputs
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract generated part
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if full_output.startswith(formatted_prompt):
                response = full_output[len(formatted_prompt):].strip()
            else:
                # Try to extract after "Assistant:" marker
                if "Assistant:" in full_output:
                    response = full_output.split("Assistant:")[-1].strip()
                else:
                    response = full_output.strip()
            
            return response
            
        except Exception as e:
            print(f"[ERROR] XuanYuan-6B generation failed: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def get_response(self, prompts_list):
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts_list: List of prompts (each is context + adversarial suffix)
        
        Returns:
            List of responses
        """
        from config import TARGET_MAX_N_TOKENS
        import time
        
        if self.experiment_logger:
            self.experiment_logger.log_stage_start("Target Model (XuanYuan-6B) Generation")
        
        # Move model to GPU for inference
        print(f"[INFO] Moving XuanYuan-6B to GPU for batch generation ({len(prompts_list)} prompts)...")
        start_time = time.time()
        self.model = self.model.to(self.gpu_device)
        move_time = time.time() - start_time
        print(f"[INFO] Model moved to GPU in {move_time:.2f}s")
        
        try:
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Format all prompts with XuanYuan conversation template
            formatted_prompts = [self._format_prompt(p) for p in prompts_list]
            
            # Batch tokenize (left padding for generation)
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(
                formatted_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.gpu_device)
            
            print(f"[INFO] Batch generating {len(prompts_list)} responses...")
            gen_start = time.time()
            
            # Batch generate (greedy decoding)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=TARGET_MAX_N_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            gen_time = time.time() - gen_start
            print(f"[INFO] Batch generation completed in {gen_time:.2f}s")
            
            # Decode outputs and extract generated parts
            responses = []
            batch_input_len = inputs['input_ids'].shape[1]
            
            for i, output in enumerate(outputs):
                generated_tokens = output[batch_input_len:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                responses.append(response)
            
            # Log first response if logger available
            if self.experiment_logger and len(responses) > 0:
                self.experiment_logger.log_model_call(
                    model_name="XuanYuan-6B (Target, Batch)",
                    parameters={
                        'max_new_tokens': TARGET_MAX_N_TOKENS,
                        'do_sample': False,
                        'decoding': 'greedy',
                        'batch_size': len(prompts_list)
                    },
                    input_prompt=formatted_prompts[0][:500] + "..." if len(formatted_prompts[0]) > 500 else formatted_prompts[0],
                    raw_output=responses[0],
                    processed_output=f"Batch size: {len(prompts_list)}, Gen time: {gen_time:.2f}s"
                )
                
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            print(f"[INFO] Falling back to sequential generation...")
            responses = []
            for idx, prompt in enumerate(prompts_list):
                response = self.generate_single_response(prompt)
                responses.append(response)
        
        finally:
            # Move model back to CPU to free GPU memory
            if not getattr(self, '_skip_offload', False):
                print(f"[INFO] Moving XuanYuan-6B back to CPU...")
                start_time = time.time()
                self.model = self.model.to("cpu")
                torch.cuda.empty_cache()
                move_time = time.time() - start_time
            print(f"[INFO] Model moved to CPU in {move_time:.2f}s, GPU memory freed")
            
            if self.experiment_logger:
                self.experiment_logger.log_stage_end("Target Model (XuanYuan-6B) Generation")
        
        return responses

# ============================================================================
# FinGPTTargetLM - FinGPT (Llama-3-8B + LoRA) for financial decision attacks
# Created: 2026-01-20
# ============================================================================
class FinGPTTargetLM:
    """
    Target model for financial decision attacks using FinGPT.
    FinGPT is based on Llama-3-8B with LoRA fine-tuning for financial sentiment analysis.
    
    Model loading: LlamaForCausalLM + PeftModel (LoRA adapter)
    No special prompt format needed - PIXIU's doc_to_text() generates compatible format.
    
    Modified: 2026-01-22 - Added validate_text() and safe tokenization to prevent CUDA errors
    """
    def __init__(self, 
                 base_model_path="../models/Meta-Llama-3-8B",
                 lora_path="../models/fingpt-mt_llama3-8b_lora",
                 experiment_logger=None):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.experiment_logger = experiment_logger
        self.model_name = "FinGPT (Llama-3-8B + LoRA)"
        
        print(f"[INFO] Loading FinGPT from {base_model_path} + {lora_path}...")
        print("[INFO] Using fp16 precision with CPU offload strategy...")
        
        # Import required modules
        from transformers import LlamaForCausalLM, LlamaTokenizerFast
        from peft import PeftModel
        
        # Load base model to CPU (CPU offload strategy)
        print("[INFO] Loading base model (Llama-3-8B)...")
        self.model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU, move to GPU when needed
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        print("[INFO] Loading LoRA adapter (fingpt-mt_llama3-8b_lora)...")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = LlamaTokenizerFast.from_pretrained(base_model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store vocab size for validation (2026-01-29)
        # Note: Use len(tokenizer) instead of vocab_size to include special tokens
        # Llama-3 has vocab_size=128000 but actual tokenizer has 128256 tokens (including eos=128001)
        self.vocab_size = len(self.tokenizer)
        print(f"[INFO] FinGPT vocab size: {self.vocab_size} (tokenizer.vocab_size={self.tokenizer.vocab_size})")
        
        self.gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] FinGPT loaded to CPU successfully")
        print(f"[INFO] Will use {self.gpu_device} for inference when needed")
    
    def validate_text(self, text: str) -> tuple:
        """
        Validate if text can be safely tokenized without producing out-of-range token IDs.
        
        Returns:
            (is_valid, reason): is_valid=True if text is safe, False otherwise
        
        Added: 2026-01-22 - Prevent CUDA device-side assert errors
        """
        try:
            # Tokenize the text
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            input_ids = tokens['input_ids']
            
            # Check for out-of-range token IDs
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            
            if max_id >= self.vocab_size:
                return False, f"Token ID {max_id} >= vocab_size {self.vocab_size}"
            if min_id < 0:
                return False, f"Token ID {min_id} < 0"
            
            return True, "OK"
        except Exception as e:
            return False, f"Tokenization error: {str(e)}"
    
    def validate_candidates(self, candidates: list, source_input: str) -> list:
        """
        Filter candidates that would cause CUDA errors.
        
        Args:
            candidates: List of candidate adversarial suffixes
            source_input: The original input text
            
        Returns:
            List of (candidate, is_valid, reason) tuples
        
        Added: 2026-01-29
        """
        results = []
        for candidate in candidates:
            full_prompt = source_input + " " + candidate
            is_valid, reason = self.validate_text(full_prompt)
            results.append((candidate, is_valid, reason))
        return results
    
    def generate_single_response(self, full_prompt):
        """
        Generate response for a single prompt.
        Uses greedy decoding to match PIXIU evaluation settings.
        
        Modified: 2026-01-29 - Added token ID clamping safety net
        """
        try:
            from config import TARGET_MAX_N_TOKENS
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            # === SAFETY NET (2026-01-29) ===
            # Clamp token IDs to valid range to prevent CUDA device-side assert
            if inputs['input_ids'].max().item() >= self.vocab_size:
                inputs['input_ids'] = inputs['input_ids'].clamp(0, self.vocab_size - 1)
            # === END SAFETY NET ===
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=TARGET_MAX_N_TOKENS,
                    min_new_tokens=2,  # FinGPT generates EOS as first token, need to skip it
                    do_sample=False,  # Greedy decoding for deterministic outputs
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only generated part
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][prompt_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            return response
            
        except Exception as e:
            print(f"[ERROR] FinGPT generation failed: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def _batched_generation(self, prompts_list, batch_size, max_new_tokens):
        """
        Generate responses in smaller batches to avoid OOM.
        
        Added: 2026-01-29 - For handling long prompts that cause OOM
        """
        all_responses = []
        total_batches = (len(prompts_list) + batch_size - 1) // batch_size
        
        print(f"[INFO] Processing {len(prompts_list)} prompts in {total_batches} batches (batch_size={batch_size})")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(prompts_list))
            batch_prompts = prompts_list[start_idx:end_idx]
            
            print(f"[INFO] Processing batch {batch_idx + 1}/{total_batches} ({len(batch_prompts)} prompts)...")
            
            try:
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.gpu_device)
                
                # Safety net: clamp token IDs
                if inputs['input_ids'].max().item() >= self.vocab_size:
                    inputs['input_ids'] = inputs['input_ids'].clamp(0, self.vocab_size - 1)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=2,  # FinGPT generates EOS as first token
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                batch_input_len = inputs['input_ids'].shape[1]
                for output in outputs:
                    generated_tokens = output[batch_input_len:]
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    all_responses.append(response)
                
                # Clear GPU cache between batches
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx + 1} failed: {e}")
                # Fall back to sequential for this batch
                for prompt in batch_prompts:
                    response = self.generate_single_response(prompt)
                    all_responses.append(response)
        
        return all_responses
    
    def get_response(self, prompts_list):
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts_list: List of prompts (each is context + adversarial suffix)
        
        Returns:
            List of responses
        
        Modified: 2026-01-29 - Added OOM handling with dynamic batch size
        """
        from config import TARGET_MAX_N_TOKENS
        import time
        
        if self.experiment_logger:
            self.experiment_logger.log_stage_start("Target Model (FinGPT) Generation")
        
        # Move model to GPU for inference
        print(f"[INFO] Moving FinGPT to GPU for batch generation ({len(prompts_list)} prompts)...")
        start_time = time.time()
        self.model = self.model.to(self.gpu_device)
        move_time = time.time() - start_time
        print(f"[INFO] Model moved to GPU in {move_time:.2f}s")
        
        try:
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # === OOM PREVENTION (2026-01-29) ===
            # Check average prompt length and adjust batch size for long prompts
            avg_len = sum(len(p) for p in prompts_list) / len(prompts_list) if prompts_list else 0
            max_len = max(len(p) for p in prompts_list) if prompts_list else 0
            
            # Dynamic batch size based on text length
            if max_len > 1500:
                effective_batch_size = 4
                print(f"[INFO] Long prompts detected (max={max_len} chars), using batch_size={effective_batch_size}")
            elif max_len > 1000:
                effective_batch_size = 8
                print(f"[INFO] Medium-long prompts detected (max={max_len} chars), using batch_size={effective_batch_size}")
            else:
                effective_batch_size = len(prompts_list)  # Full batch
            # === END OOM PREVENTION ===
            
            # Process in batches if needed
            if effective_batch_size < len(prompts_list):
                return self._batched_generation(prompts_list, effective_batch_size, TARGET_MAX_N_TOKENS)
            
            # Batch tokenize (left padding for generation)
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(
                prompts_list, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.gpu_device)
            
            # === SAFETY NET (2026-01-29) ===
            # Clamp token IDs to valid range to prevent CUDA device-side assert
            original_max = inputs['input_ids'].max().item()
            if original_max >= self.vocab_size:
                print(f"[WARN] Detected out-of-range token ID: {original_max} >= {self.vocab_size}")
                print(f"[WARN] Clamping to valid range [0, {self.vocab_size - 1}]")
                inputs['input_ids'] = inputs['input_ids'].clamp(0, self.vocab_size - 1)
            # === END SAFETY NET ===
            
            print(f"[INFO] Batch generating {len(prompts_list)} responses...")
            gen_start = time.time()
            
            # Batch generate (greedy decoding)
            # min_new_tokens=2: FinGPT generates EOS as first token, need to skip it
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=TARGET_MAX_N_TOKENS,
                    min_new_tokens=2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            gen_time = time.time() - gen_start
            print(f"[INFO] Batch generation completed in {gen_time:.2f}s")
            
            # Decode outputs and extract generated parts
            responses = []
            batch_input_len = inputs['input_ids'].shape[1]
            
            for i, output in enumerate(outputs):
                generated_tokens = output[batch_input_len:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                responses.append(response)
            
            # Log first response if logger available
            if self.experiment_logger and len(responses) > 0:
                self.experiment_logger.log_model_call(
                    model_name="FinGPT (Target, Batch)",
                    parameters={
                        'max_new_tokens': TARGET_MAX_N_TOKENS,
                        'do_sample': False,
                        'decoding': 'greedy',
                        'batch_size': len(prompts_list)
                    },
                    input_prompt=prompts_list[0][:500] + "..." if len(prompts_list[0]) > 500 else prompts_list[0],
                    raw_output=responses[0],
                    processed_output=f"Batch size: {len(prompts_list)}, Gen time: {gen_time:.2f}s"
                )
                
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            print(f"[INFO] Falling back to sequential generation...")
            responses = []
            for idx, prompt in enumerate(prompts_list):
                response = self.generate_single_response(prompt)
                responses.append(response)
        
        finally:
            # Move model back to CPU to free GPU memory
            if not getattr(self, '_skip_offload', False):
                print(f"[INFO] Moving FinGPT back to CPU...")
                start_time = time.time()
                self.model = self.model.to("cpu")
                torch.cuda.empty_cache()
                move_time = time.time() - start_time
            print(f"[INFO] Model moved to CPU in {move_time:.2f}s, GPU memory freed")
            
            if self.experiment_logger:
                self.experiment_logger.log_stage_end("Target Model (FinGPT) Generation")
        
        return responses

# ============================================================================
# FinR1TargetLM - Fin-R1 for financial decision attacks
# Created: 2026-02-06
# ============================================================================
class FinR1TargetLM:
    """
    Target model for financial decision attacks using Fin-R1.
    Fin-R1 is based on Qwen2.5-7B-Instruct, outputs <think>...</think><answer>...</answer> format.
    
    Key: Strip <think> part and extract only the <answer> content for classification.
    """
    def __init__(self, model_path="../models/Fin-R1", experiment_logger=None):
        import re
        self.model_path = model_path
        self.experiment_logger = experiment_logger
        self.model_name = "Fin-R1"
        self._think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
        
        print(f"[INFO] Loading Fin-R1 from {model_path}...")
        print("[INFO] Using bfloat16 precision with CPU offload strategy...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load to CPU, move to GPU when needed
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Fin-R1 loaded to CPU successfully")
        print(f"[INFO] Will use {self.gpu_device} for inference when needed")
    
    def _strip_thinking(self, text):
        """
        Remove <think>...</think> part from Fin-R1 output.
        Also extract the final answer from various formats:
        1. <answer>...</answer> tags
        2. \\boxed{...} format (LaTeX-style, common in Qwen2.5 outputs)
        3. "Final Answer: ..." pattern at the end
        
        Modified: 2026-02-07 - Added \\boxed{} and "Final Answer:" extraction
        """
        # Step 1: Try to extract content between <answer> tags
        if '<answer>' in text and '</answer>' in text:
            start = text.find('<answer>') + len('<answer>')
            end = text.find('</answer>')
            return text[start:end].strip()
        
        # Step 2: Remove <think>...</think> and return the rest
        cleaned = self._think_pattern.sub('', text).strip()
        
        # If there's still <answer> tag without closing, extract after it
        if '<answer>' in cleaned:
            cleaned = cleaned.split('<answer>')[-1].strip()
            return cleaned
        
        # Step 3: Try to extract from \boxed{...} (Qwen2.5 reasoning format)
        import re
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', cleaned)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Step 4: Try "Answer:" / "Final Answer:" pattern (last occurrence)
        # Remove markdown bold markers (**) for cleaner matching
        clean_for_match = cleaned.replace('**', '')
        answer_matches = list(re.finditer(
            r'(?:Final\s+)?Answer\s*[:\-]\s*(.+?)$',
            clean_for_match, re.MULTILINE | re.IGNORECASE
        ))
        if answer_matches:
            return answer_matches[-1].group(1).strip()
        
        return cleaned
    
    def _format_prompt(self, content):
        """
        Format content using Fin-R1's chat template.
        Uses Qwen2.5-Instruct style with system prompt.
        """
        messages = [
            {"role": "user", "content": content}
        ]
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    # 2026-02-07 - Fin-R1 needs much larger max_new_tokens because it generates
    # <think>...</think><answer>...</answer> format. The thinking chain can be 200-500 tokens.
    FINR1_MAX_NEW_TOKENS = 1024
    
    def generate_single_response(self, full_prompt):
        """
        Generate response for a single prompt.
        Uses greedy decoding for deterministic outputs.
        """
        try:
            formatted_prompt = self._format_prompt(full_prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.FINR1_MAX_NEW_TOKENS,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            new_tokens = outputs[0][prompt_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Strip <think> part and extract answer
            response = self._strip_thinking(response)
            
            return response
            
        except Exception as e:
            print(f"[ERROR] Fin-R1 generation failed: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def get_response(self, prompts_list):
        """
        Generate responses for a batch of prompts.
        """
        from config import TARGET_MAX_N_TOKENS
        import time
        
        if self.experiment_logger:
            self.experiment_logger.log_stage_start("Target Model (Fin-R1) Generation")
        
        # Move model to GPU for inference
        print(f"[INFO] Moving Fin-R1 to GPU for batch generation ({len(prompts_list)} prompts)...")
        start_time = time.time()
        self.model = self.model.to(self.gpu_device)
        move_time = time.time() - start_time
        print(f"[INFO] Model moved to GPU in {move_time:.2f}s")
        
        try:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Format all prompts with Fin-R1 chat template
            formatted_prompts = [self._format_prompt(p) for p in prompts_list]
            
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(
                formatted_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.gpu_device)
            
            print(f"[INFO] Batch generating {len(prompts_list)} responses...")
            gen_start = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.FINR1_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            gen_time = time.time() - gen_start
            print(f"[INFO] Batch generation completed in {gen_time:.2f}s")
            
            responses = []
            batch_input_len = inputs['input_ids'].shape[1]
            
            for i, output in enumerate(outputs):
                generated_tokens = output[batch_input_len:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                # Strip <think> part and extract answer
                response = self._strip_thinking(response)
                responses.append(response)
            
            if self.experiment_logger and len(responses) > 0:
                self.experiment_logger.log_model_call(
                    model_name="Fin-R1 (Target, Batch)",
                    parameters={
                        'max_new_tokens': self.FINR1_MAX_NEW_TOKENS,
                        'do_sample': False,
                        'decoding': 'greedy',
                        'batch_size': len(prompts_list)
                    },
                    input_prompt=formatted_prompts[0][:500] + "...",
                    raw_output=responses[0],
                    processed_output=f"Batch size: {len(prompts_list)}, Gen time: {gen_time:.2f}s"
                )
                
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            print(f"[INFO] Falling back to sequential generation...")
            responses = []
            for idx, prompt in enumerate(prompts_list):
                response = self.generate_single_response(prompt)
                responses.append(response)
        
        finally:
            # Move model back to CPU to free GPU memory
            if not getattr(self, '_skip_offload', False):
                print(f"[INFO] Moving Fin-R1 back to CPU...")
                start_time = time.time()
                self.model = self.model.to("cpu")
                torch.cuda.empty_cache()
                move_time = time.time() - start_time
            print(f"[INFO] Model moved to CPU in {move_time:.2f}s, GPU memory freed")
            
            if self.experiment_logger:
                self.experiment_logger.log_stage_end("Target Model (Fin-R1) Generation")
        
        return responses