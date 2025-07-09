import os
os.sys.path.append("..")
from configs.template import get_config as default_config
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle

MODEL_CONFIGS = {
    "vicuna-7b": {
        "path": "lmsys/vicuna-7b-v1.5",
        "template": "vicuna",
        "use_fast": False
    },
    "vicuna-13b": {
        "path": "lmsys/vicuna-13b-v1.5", 
        "template": "vicuna",
        "use_fast": False
    },
    "vicuna-33b": {
        "path": "lmsys/vicuna-33b-v1.3",
        "template": "vicuna", 
        "use_fast": False
    },
    "llama2-7b": {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "template": "llama-2",
        "use_fast": False
    },
    "llama2-13b": {
        "path": "meta-llama/Llama-2-13b-chat-hf",
        "template": "llama-2",
        "use_fast": False
    },
    "llama2-70b": {
        "path": "meta-llama/Llama-2-70b-chat-hf",
        "template": "llama-2", 
        "use_fast": False
    },
    "mistral-7b": {
        "path": "mistralai/Mistral-7B-Instruct-v0.2",
        "template": "mistral",
        "use_fast": False,
        "custom_template": True
    },
    "guanaco-7b": {
        "path": "TheBloke/guanaco-7B-HF",
        "template": "guanaco",
        "use_fast": False
    },
    "guanaco-13b": {
        "path": "TheBloke/guanaco-13B-HF", 
        "template": "guanaco",
        "use_fast": False
    },
    "pythia-12b": {
        "path": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "template": "oasst_pythia",
        "use_fast": True
    }
}

def get_config(model_name, transfer=False, device="cuda:0"):
    config = default_config()
    
    if transfer:
        config.transfer = True
        config.logfile = ""
        config.progressive_goals = False
        config.stop_on_success = False
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = MODEL_CONFIGS[model_name]
    
    config.tokenizer_paths = [model_config["path"]]
    config.tokenizer_kwargs = [{"use_fast": model_config["use_fast"]}]
    config.model_paths = [model_config["path"]]
    config.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates = [model_config["template"]]
    config.devices = [device]
    
    # Handle special cases
    if model_name in ["vicuna-7b", "vicuna-13b", "vicuna-33b", "llama2-7b", "llama2-13b", "llama2-70b", "mistral-7b"]:
        config.incr_control = True
    
    if model_name == "mistral-7b":
        register_conv_template(
            Conversation(
                name="mistral",
                system_template="[INST] {system_message}\n",
                roles=("[INST]", "[/INST]"),
                sep_style=SeparatorStyle.LLAMA2,
                sep=" ",
                sep2="</s>",
            ),
            override=True
        )
    
    return config
