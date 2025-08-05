from advsecurenet.llm_RLHF_finetuning.config import DataConfig, ModelConfig, TrainingConfig
from advsecurenet.llm_RLHF_finetuning.model_loader import ModelLoader
from advsecurenet.llm_RLHF_finetuning.dataloader import DataLoader
from advsecurenet.llm_RLHF_finetuning.trainer import FineTuningPipeline

# Using hardocded config
data_config = DataConfig(dataset_name="trl-lib/Capybara")
model_config = ModelConfig(model_name_or_path="gpt2", use_peft=True)
training_config = TrainingConfig(output_dir="./gpt2-sft")

# Instantiate pipeline
data_loader = DataLoader(data_config)
model_loader = ModelLoader(model_config)
pipeline = FineTuningPipeline(model_loader, data_loader, training_config)

# Run fine-tuning
pipeline.run()
