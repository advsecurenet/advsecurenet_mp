from advsecurenet.llm_finetuning.config_manager import ConfigManager
from advsecurenet.llm_finetuning.model_loader import ModelLoader
from advsecurenet.llm_finetuning.data_loader import HuggingFaceDataLoader
from advsecurenet.llm_finetuning.enhanced_trainer import Trainer
from advsecurenet.llm_finetuning.metrics import MetricsCalculator
from typing import List, Optional

class ClassificationPipeline:
    """Pipeline specifically for text classification tasks."""
    
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
    
    def run(self, use_wandb: bool = False, wandb_project: str = None, early_stopping: bool = True):
        """Run the classification training pipeline."""
        
        # Load model and tokenizer
        model_loader = ModelLoader(self.config_manager.model_config)
        model, tokenizer = model_loader.load_model_and_tokenizer()
        
        # Load and prepare data for classification
        data_loader = HuggingFaceDataLoader(tokenizer, self.config_manager.data_config)
        train_dataset, val_dataset = data_loader.load_data(
            dataset_name=self.config_manager.data_config.dataset_name,
            text_column=self.config_manager.data_config.text_column,
            label_column=self.config_manager.data_config.label_column,
            validation_split=self.config_manager.data_config.validation_split,
            max_length=self.config_manager.data_config.max_length,
            preprocessing_num_workers=self.config_manager.data_config.preprocessing_num_workers
        )
        
        # Setup metrics for classification
        metrics_calculator = MetricsCalculator()
        
        # Train model
        trainer = Trainer(model, tokenizer, self.config_manager, metrics_calculator)
        results = trainer.train(
            train_dataset, 
            val_dataset,
            early_stopping=early_stopping
        )
        
        return results