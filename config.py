from enum import Enum
VICUNA_PATH = "../vicuna-13b-v1.5"
LLAMA_PATH = "../Llama-2-7b-chat-hf"

ATTACK_TEMP = 1
TARGET_TEMP = 0  
ATTACK_TOP_P = 0.9
TARGET_MAX_N_TOKENS = 80  # Modified: 2026-02-02 - Increased from 50 to avoid truncation-induced false positives  
ATTACK_MAX_N_TOKENS = 30  


## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna-13b-v1.5"
    llama_2 = "llama-2-7b-chat-hf"
    llama_3_1 = "llama-3.1-8b"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4-0125-preview"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    gemini = "gemini-pro"
    mixtral = "mixtral"

MODEL_NAMES = [model.value for model in Model]


HF_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "../Llama-2-7b-chat-hf",
    Model.llama_3_1: "../models/Llama-3.1-8B",  # Used for classification attack (2025-11-25)
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
    Model.mixtral: "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

TOGETHER_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "together_ai/togethercomputer/llama-2-7b-chat",
    Model.vicuna: "together_ai/lmsys/vicuna-13b-v1.5",
    Model.mixtral: "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"
}

FASTCHAT_TEMPLATE_NAMES: dict[Model, str] = {
    Model.gpt_3_5: "gpt-3.5-turbo",
    Model.gpt_4: "gpt-4",
    Model.claude_1: "claude-instant-1.2",
    Model.claude_2: "claude-2.1",
    Model.gemini: "gemini-pro",
    Model.vicuna: "vicuna_v1.1",
    Model.llama_2: "llama-2-7b-chat-hf",
    Model.llama_3_1: "llama-3.1-8b",
    Model.mixtral: "mixtral",
}

API_KEY_NAMES: dict[Model, str] = {
    Model.gpt_3_5:  "OPENAI_API_KEY",
    Model.gpt_4:    "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.gemini:   "GEMINI_API_KEY",
    Model.vicuna:   "TOGETHER_API_KEY",
    Model.llama_2:  "TOGETHER_API_KEY",
    Model.mixtral:  "TOGETHER_API_KEY",
}

# 自定义 API 配置（用于第三方 API 服务）
# 环境变量名，用于从环境变量读取 API key 和 base URL
CUSTOM_API_KEY_ENV = "CUSTOM_API_KEY"  # 自定义 API 的 key 环境变量名
CUSTOM_API_BASE_ENV = "CUSTOM_API_BASE"  # 自定义 API 的 base URL 环境变量名

LITELLM_TEMPLATES: dict[Model, dict] = {
    Model.vicuna: {"roles":{
                    "system": {"pre_message": "", "post_message": " "},
                    "user": {"pre_message": "USER: ", "post_message": " ASSISTANT:"},
                    "assistant": {
                        "pre_message": "",
                        "post_message": "",
                    },
                },
                "post_message":"</s>",
                "initial_prompt_value" : "",
                "eos_tokens": ["</s>"]         
                },
    Model.llama_2: {"roles":{
                    "system": {"pre_message": "[INST] <<SYS>>\n", "post_message": "\n<</SYS>>\n\n"},
                    "user": {"pre_message": "", "post_message": " [/INST]"},
                    "assistant": {"pre_message": "", "post_message": ""},
                },
                "post_message" : " </s><s>",
                "initial_prompt_value" : "",
                "eos_tokens" :  ["</s>", "[/INST]"]  
            },
    Model.mixtral: {"roles":{
                    "system": {
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    },
                    "user": { 
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    }, 
                    "assistant": {
                        "pre_message": " ",
                        "post_message": "",
                    }
                },
                "post_message": "</s>",
                "initial_prompt_value" : "<s>",
                "eos_tokens": ["</s>", "[/INST]"]
    },
    Model.llama_3_1: {"roles":{
                    "system": {
                        "pre_message": "<|start_header_id|>system<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>"
                    },
                    "user": {
                        "pre_message": "<|start_header_id|>user<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>"
                    },
                    "assistant": {
                        "pre_message": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>"
                    }
                },
                "post_message": "",
                "initial_prompt_value": "<|begin_of_text|>",
                "eos_tokens": ["<|eot_id|>", "<|end_of_text|>"]
    }
}