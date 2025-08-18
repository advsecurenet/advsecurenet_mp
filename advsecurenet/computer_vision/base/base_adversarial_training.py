import logging
import random
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.computer_vision.image_classification.attacks import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import (
    AdversarialTrainingConfig,
)
from advsecurenet.trainer.trainer import Trainer

class BaseAdversarialTraining(Trainer):
    """
    Base class for adversarial training methods.
    """

    def __init__(self, config: AdversarialTrainingConfig) -> None:
        self.config: AdversarialTrainingConfig = config
        super().__init__(config)
    
    def _check_config_base(self, config: AdversarialTrainingConfig) -> None:
        # Check configuration validity
        if not isinstance(config.model, BaseModel):
            raise ValueError("Target model must be a subclass of BaseModel!")
        if not all(isinstance(model, BaseModel) for model in config.models):
            raise ValueError("All models must be a subclass of BaseModel!")
        if not all(isinstance(attack, AdversarialAttack) for attack in config.attacks):
            raise ValueError("All attacks must be a subclass of AdversarialAttack!")
        if not isinstance(config.train_loader, DataLoader):
            raise ValueError("train_dataloader must be a DataLoader!")
        
    def _pre_training(self):
        # add target model to list of models if not already present
        if self.config.model not in self.config.models:
            self.config.models.append(self.config.model)

        # set each model to train mode
        self.config.models = [model.train() for model in self.config.models]

        # move each model to device
        self.config.models = [model.to(self._device) for model in self.config.models]

    def _get_train_loader(self, epoch: int):
        return tqdm(
            self.config.train_loader,
            desc="Adversarial Training",
            leave=False,
            position=1,
            unit="batch",
            colour="blue",
        )
    
    def _get_loss_divisor(self):
        return len(self.config.train_loader)
