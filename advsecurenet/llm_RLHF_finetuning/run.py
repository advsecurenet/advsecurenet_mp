from advsecurenet.llm_RLHF_finetuning.config import DataConfig, ModelConfig, TrainingConfig
from advsecurenet.llm_RLHF_finetuning.model_loader import ModelLoader
from advsecurenet.llm_RLHF_finetuning.dataloader import DataLoader
from advsecurenet.llm_RLHF_finetuning.trainer import FineTuningPipeline
from advsecurenet.llm_RLHF_finetuning.config import DataConfig, ModelConfig


# Using hardocded config
data_config = DataConfig(dataset_name="trl-lib/Capybara")
model_config = ModelConfig(model_name_or_path="Qwen/Qwen1.5-0.5B", use_peft=True)
training_config = TrainingConfig(output_dir="./Qwen/Qwen2-0.5B")

# Instantiate pipeline
data_loader = DataLoader(data_config, "Qwen/Qwen1.5-0.5B")
model_loader = ModelLoader(model_config)
pipeline = FineTuningPipeline(model_loader, data_loader, training_config)

# Run fine-tuning
pipeline.run()
