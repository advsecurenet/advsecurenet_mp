from transformers import Trainer, TrainerCallback, EarlyStoppingCallback
from transformers.integrations import WandbCallback
import torch
from typing import Optional, Dict, Any
import logging
import os
from pathlib import Path


class Trainer:
    def __init__(self, model, tokenizer, config_manager, metrics_calculator):
        self.model = model
        self.tokenizer = tokenizer
        self.config_manager = config_manager
        self.metrics_calculator = metrics_calculator
        
    
    def train(self, train_dataset, eval_dataset=None, use_wandb=False, 
              wandb_project=None, early_stopping=True, patience=3):
        """Enhanced training with callbacks and monitoring."""
        
        training_args = self.config_manager.get_training_arguments()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.metrics_calculator.compute_metrics if self.metrics_calculator else None
        )
        
        # Train model
        train_result = trainer.train()
        
        # Save model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Final evaluation
        if eval_dataset:
            eval_result = trainer.evaluate()
            
            # Generate detailed report
            predictions = trainer.predict(eval_dataset)
            y_pred = predictions.predictions.argmax(axis=1)
            y_true = predictions.label_ids
            
            # Save classification report
            if self.metrics_calculator:
                report = self.metrics_calculator.generate_classification_report(y_true, y_pred)
                with open(os.path.join(training_args.output_dir, "classification_report.txt"), "w") as f:
                    f.write(report)
                
                # Save confusion matrix
                try:
                    self.metrics_calculator.plot_confusion_matrix(
                        y_true, y_pred, 
                        save_path=os.path.join(training_args.output_dir, "confusion_matrix.png")
                    )
                except ImportError:
                    print("Warning: matplotlib/seaborn not available. Skipping confusion matrix plot.")
        
        return train_result
