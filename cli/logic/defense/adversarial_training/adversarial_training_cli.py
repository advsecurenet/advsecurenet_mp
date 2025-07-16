from dataclasses import asdict
from typing import List

import click

from advsecurenet.computer_vision.base.adversarial_attack import AdversarialAttack
from advsecurenet.computer_vision.image_classification.defenses.adversarial_training import (
    AdversarialTraining,
)
from advsecurenet.computer_vision.image_classification.defenses.ddp_adversarial_training import (
    DDPAdversarialTraining,
)
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig
from advsecurenet.shared.types.configs.configs import ConfigType
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import (
    AdversarialTrainingConfig,
)
from advsecurenet.utils.adversarial_target_generator import AdversarialTargetGenerator
from cli.logic.train.trainer import CLITrainer
from cli.shared.types.defense.adversarial_training import ATCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType
from cli.shared.utils.attack_mappings import attack_cli_mapping, attack_mapping
from cli.shared.utils.config import load_and_instantiate_config
from cli.shared.utils.model import create_model


class ATCLITrainer(CLITrainer):
    """
    Adversarial Training CLI class.

    This class is used to run adversarial training from the CLI.
    """

    def __init__(self, config: ATCliConfigType):
        super().__init__(config.training)
        self.at_config = config.adversarial_training
        self.adversarial_target_generator = AdversarialTargetGenerator()

    def train(self):
        """
        Public method to run adversarial training.
        """
        click.secho("Starting Adversarial Training", fg="green")

        if self.config.device.use_ddp:
            self._execute_ddp_training()
        else:
            self._execute_training()

    def _prepare_attacks(self) -> list[AdversarialAttack]:
        """
        Get the attack objects based on the attack names.

        Returns:
            list[AttackWithConfigDict]: List of attack objects.
        """
        attacks = []
        for attack_dict in self.at_config.attacks:
            for key, value in attack_dict.items():
                attack_name = key.upper()
                attack_config_path = value
                attack_config: AttackConfig = load_and_instantiate_config(
                    config=attack_config_path,
                    default_config_file=f"{attack_name.lower()}_attack_base_config.yml",
                    config_type=ConfigType.ATTACK,
                    config_class=attack_mapping[attack_name],
                )

                attack_config.device = self.config.device
                attack_type, _ = attack_cli_mapping[attack_name]

                attack_class = attack_type.value
                attack = attack_class(attack_config)

                attacks.append(attack)
        return attacks

    def _prepare_models(self) -> List[BaseModel]:
        """
        Prepare the models that will be used to generate adversarial examples.

        Returns:
            List[BaseModel]: List of initialized models based on the configuration provided.
        """
        # Early exit if no model configurations are provided
        if not self.at_config.models:
            return []

        # Load and initialize each model based on its configuration
        models = [
            create_model(
                load_and_instantiate_config(
                    config=model_config.get("config"),
                    default_config_file="model_config.yml",
                    config_type=ConfigType.MODEL,
                    config_class=ModelCliConfigType,
                )
            )
            for model_config in self.at_config.models
        ]

        return models

    def _prepare_training_environment(self) -> AdversarialTrainingConfig:

        # configure the model that will be adversarially trained
        model = self._initialize_model()

        train_loader = self._prepare_dataloader()
        train_config = self._prepare_train_config(model, train_loader)

        attacks = self._prepare_attacks()
        models = self._prepare_models()

        # add the target model to the list of models since it will be also used for generating adversarial examples
        models.append(model)

        config = AdversarialTrainingConfig(
            models=models, attacks=attacks, train_config=train_config
            )
        return config

    def _ddp_training_fn(self, rank: int, world_size: int):
        """
        This function is used to run adversarial training using DDP.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The number of processes to spawn.
        """
        # the model must be initialized in each process

        config = self._prepare_training_environment()

        ddp_trainer = DDPAdversarialTraining(config, rank, world_size)
        self._execute_static_train(ddp_trainer, config)

    def _execute_training(self) -> None:
        """
        Execute adversarial training.

        Args:
            dataset_name (str): Name of the dataset.
            attacks (list[AdversarialAttack]): List of attacks.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        config = self._prepare_training_environment()

        adversarial_training = AdversarialTraining(config)
        self._execute_static_train(adversarial_training, config)


    @staticmethod
    def _execute_static_train(trainer_instance, config) -> None:
        """
        Calls the static Trainer.train method with arguments unpacked from the trainer instance and its config.
        This function's job is to map the config fields to the train method's arguments.

        Args:
            trainer_instance: An initialized trainer instance (e.g., AdversarialTraining).
        """
        train_config = config.train_config
        
        # This assumes AdversarialTraining inherits from Trainer, so we can call train on its type.
        type(trainer_instance).train(
            epochs=train_config.training_hyperparameters_config.epochs,
            start_epoch=trainer_instance.start_epoch,
            optimizer=trainer_instance.optimizer,
            model=trainer_instance.model,
            train_loader=trainer_instance.train_loader,
            device=trainer_instance.device,
            loss_fn=trainer_instance.loss_fn,
            scheduler=trainer_instance.scheduler,
            save_checkpoint=train_config.checkpoint_config.save_checkpoint,
            checkpoint_interval=train_config.checkpoint_config.checkpoint_interval,
            save_final_model=train_config.final_model_config.save_final_model,
            privacy_engine=trainer_instance.privacy_engine,
            # Pass the config objects for path/name generation inside the static method
            checkpoint_config=train_config.checkpoint_config,
            final_model_config=train_config.final_model_config,
            # Pass model/dataset names if they exist on the config
            model_name=getattr(train_config.model_config, "name", "model"),
            dataset_name=getattr(train_config.training_process_config, "dataset_name", "dataset"),
        )