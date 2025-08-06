from trl import SFTTrainer
from transformers import TrainingArguments
from advsecurenet.llm_RLHF_finetuning.config import TrainingConfig
from advsecurenet.llm_RLHF_finetuning.model_loader import ModelLoader
from advsecurenet.llm_RLHF_finetuning.dataloader import DataLoader
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    clone_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

class FineTuningPipeline:
    def __init__(self, model_loader: ModelLoader, data_loader: DataLoader, training_config: TrainingConfig):
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.training_config = training_config

    def run(self):
        model, tokenizer = self.model_loader.load()
        train_data, eval_data = self.data_loader.load()

        args = SFTConfig(
            output_dir=self.training_config.output_dir,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            num_train_epochs=self.training_config.num_train_epochs,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            fp16=self.training_config.fp16,
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,
            gradient_checkpointing=True,
            eval_steps=self.training_config.eval_steps,
            push_to_hub=self.training_config.push_to_hub,
            packing=False,
        )
        if tokenizer.chat_template is None:
            result = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")
            model, tokenizer = result[0], result[1]



        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            processing_class=tokenizer,
            peft_config=self.model_loader.get_peft_config()
         )

        trainer.train()
        trainer.save_model(self.training_config.output_dir)
        if self.training_config.push_to_hub:
            trainer.push_to_hub()
