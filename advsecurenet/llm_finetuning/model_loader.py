from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple, Any
import torch

class ModelLoader:
    def __init__(self, config):
        self.config = config

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer based on configuration."""
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model based on configuration type
        if hasattr(self.config, 'model_obj') and self.config.model_obj is not None:
            # Use custom model object
            model = self.config.model_obj
        elif hasattr(self.config, 'checkpoint_path') and self.config.checkpoint_path:
            # Load from checkpoint
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.checkpoint_path,
                num_labels=self.config.num_labels
            )
        else:
            # Load from HuggingFace Hub
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )

        return model, tokenizer

    def save_model_and_tokenizer(self, model, tokenizer, save_path: str):
        """Save model and tokenizer to specified path."""
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
    def load_model_from_checkpoint(self, checkpoint_path: str) -> Tuple[Any, Any]:
        """Load model and tokenizer from a saved checkpoint."""
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
