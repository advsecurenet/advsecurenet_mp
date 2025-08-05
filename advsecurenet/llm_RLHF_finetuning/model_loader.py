from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from advsecurenet.llm_RLHF_finetuning.config import ModelConfig
from trl import get_peft_config, get_quantization_config, get_kbit_device_map

class ModelLoader:
    def __init__(self, config: ModelConfig):
        self.config = config

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, trust_remote_code=self.config.trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token exists

        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": "auto",
            "use_cache": not self.config.gradient_checkpointing,
        }

        if self.config.quantize:
            model_kwargs["quantization_config"] = get_quantization_config(self.config)
            model_kwargs["device_map"] = get_kbit_device_map()

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, **model_kwargs)
        return model, tokenizer

    def get_peft_config(self):
        if self.config.use_peft:
            return get_peft_config(self.config)
        return None
