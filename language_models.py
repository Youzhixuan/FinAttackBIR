# 自己写的
import os 
import litellm
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model, CUSTOM_API_KEY_ENV, CUSTOM_API_BASE_ENV
from loggers import logger
from common import get_api_key

# === START: Add this new section for your custom API 自己加的 ===
import openai
import time

def query_custom_api(conversation_messages, model_name, max_tokens, temperature, top_p, api_key, api_base):
    # Set API credentials
    openai.api_key = api_key
    openai.api_base = api_base

    try:
        # Note: We pass through parameters like max_tokens and temperature.
        # Your API provider must support these for them to take effect.
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=conversation_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        # Using the logger from the original file for consistent logging
        logger.error(f"Custom API call failed for model {model_name}: {e}")
        return "ERROR: API CALL FAILED."
# === END: Add this new section for your custom API 自己加的 ===

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        else:
            self.use_open_source_model =  False
            #if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value 
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    
    
    # === START: Replace the old batched_generate method with this new one ===
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 

        # 从环境变量读取自定义 API 配置
        try:
            custom_api_key = os.environ[CUSTOM_API_KEY_ENV]
        except KeyError:
            raise ValueError(
                f"Missing custom API key. Please set the environment variable {CUSTOM_API_KEY_ENV}. "
                f"You can set it by running: export {CUSTOM_API_KEY_ENV}='your-api-key-here'"
            )
        
        try:
            custom_api_base = os.environ[CUSTOM_API_BASE_ENV]
        except KeyError:
            raise ValueError(
                f"Missing custom API base URL. Please set the environment variable {CUSTOM_API_BASE_ENV}. "
                f"You can set it by running: export {CUSTOM_API_BASE_ENV}='https://api.example.com/v1'"
            )

        responses = []
        for conv in convs_list:
            # self.litellm_model_name is correctly derived from config.py (e.g., "gpt-3.5-turbo-1106")
            # We pass it to your custom API function.
            response_content = query_custom_api(
                conversation_messages=conv,
                model_name=self.model_name.value, # Use the model name from the enum, e.g., "gpt-3.5-turbo-1106"
                max_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                api_key=custom_api_key,
                api_base=custom_api_base
            )
            responses.append(response_content)
            # Respect the original code's sleep interval between queries
            time.sleep(self.API_QUERY_SLEEP)

        return responses
    # === END: Replace the old batched_generate method with this new one ===

# class LocalvLLM(LanguageModel):
    
#     def __init__(self, model_name: str):
#         """Initializes the LLMHuggingFace with the specified model name."""
#         super().__init__(model_name)
#         if self.model_name not in MODEL_NAMES:
#             raise ValueError(f"Invalid model name: {model_name}")
#         self.hf_model_name = HF_MODEL_NAMES[Model(model_name)]
#         destroy_model_parallel()
#         self.model = vllm.LLM(model=self.hf_model_name)
#         if self.temperature > 0:
#             self.sampling_params = vllm.SamplingParams(
#                 temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_n_tokens
#             )
#         else:
#             self.sampling_params = vllm.SamplingParams(temperature=0, max_tokens=self.max_n_tokens)

#     def _get_responses(self, prompts_list: list[str], max_new_tokens: int | None = None) -> list[str]:
#         """Generates responses from the model for the given prompts."""
#         full_prompt_list = self._prompt_to_conv(prompts_list)
#         outputs = self.model.generate(full_prompt_list, self.sampling_params)
#         # Get output from each input, but remove initial space
#         outputs_list = [output.outputs[0].text[1:] for output in outputs]
#         return outputs_list

#     def _prompt_to_conv(self, prompts_list):
#         batchsize = len(prompts_list)
#         convs_list = [self._init_conv_template() for _ in range(batchsize)]
#         full_prompts = []
#         for conv, prompt in zip(convs_list, prompts_list):
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], None)
#             full_prompt = conv.get_prompt()
#             # Need this to avoid extraneous space in generation
#             if "llama-2-7b-chat-hf" in self.model_name:
#                 full_prompt += " "
#             full_prompts.append(full_prompt)
#         return full_prompts

#     def _init_conv_template(self):
#         template = get_conversation_template(self.hf_model_name)
#         if "llama" in self.hf_model_name:
#             # Add the system prompt for Llama as FastChat does not include it
#             template.system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#         return template
    






