from typing import List, Optional

import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator


class TransferabilityEvaluator(BaseEvaluator):
    """
    Evaluates the transferability of adversarial examples generated by a source model to a list of target models.

    Args:
        target_models (List): List of target models to evaluate the transferability of adversarial examples to.

    Attributes:
        transferability_data (dict): Dictionary containing the transferability data for each target model.

    """

    def __init__(self, target_models: List):
        self.target_models = target_models
        self.reset()

    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        self.transferability_data = {}
        # if we have multiple models of the same type, we append a number to the model name starting from 2
        model_names_dict = {}
        for model in self.target_models:
            model_name = model.model_name
            if model_name not in model_names_dict:
                model_names_dict[model_name] = 1
            else:
                model_names_dict[model_name] += 1
            model_name += f"_{model_names_dict[model_name]}"
            self.transferability_data[model_name] = {
                'successful_transfer': 0, 'successful_on_source': 0}

        self.total_successful_on_source = 0

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # def update(self, source_model, original_images: torch.Tensor, true_labels: torch.Tensor, adversarial_images: torch.Tensor, is_targeted: bool = False, target_labels: Optional[torch.Tensor] = None):
    #     source_model.eval()
    #     source_predictions = source_model(adversarial_images)
    #     source_labels = torch.argmax(source_predictions, dim=1)

    #     if is_targeted and target_labels is None:
    #         raise ValueError(
    #             "Target labels must be provided for targeted attacks.")

    #     successful_on_source = torch.sum(
    #         source_labels == target_labels if is_targeted else source_labels != true_labels)
    #     self.total_successful_on_source += successful_on_source
    #     model_names_dict = {}
    #     for target_model in self.target_models:
    #         model_name = target_model.model_name
    #         if model_name not in model_names_dict:
    #             model_names_dict[model_name] = 1
    #         else:
    #             model_names_dict[model_name] += 1
    #         model_name += f"_{model_names_dict[model_name]}"

    #         target_model.eval()
    #         target_predictions = target_model(adversarial_images)
    #         target_labels = torch.argmax(target_predictions, dim=1)
    #         successful_transfer = torch.sum((source_labels == target_labels) & (
    #             target_labels == true_labels) if is_targeted else (source_labels != true_labels) & (target_labels != true_labels))
    #         self.transferability_data[model_name]['successful_transfer'] += successful_transfer
    #         self.transferability_data[model_name]['successful_on_source'] += successful_on_source

    def update(self, source_model, original_images: torch.Tensor, true_labels: torch.Tensor, adversarial_images: torch.Tensor, is_targeted: bool = False, target_labels: Optional[torch.Tensor] = None):
        source_model.eval()
        source_predictions = source_model(adversarial_images)
        source_labels = torch.argmax(source_predictions, dim=1)

        if is_targeted and target_labels is None:
            raise ValueError(
                "Target labels must be provided for targeted attacks.")

        # Calculate which adversarial examples were successful on the source model
        successful_on_source_mask = (source_labels == target_labels) if is_targeted else (
            source_labels != true_labels)
        self.total_successful_on_source += successful_on_source_mask.sum().item()

        model_names_dict = {}
        # Loop through target models to evaluate transferability for each
        for target_model in self.target_models:
            # we might have multiple models of the same type, so we append a number to the model name i.e. "resnet50_2"
            model_name = target_model.model_name
            if model_name not in model_names_dict:
                model_names_dict[model_name] = 1
            else:
                model_names_dict[model_name] += 1
            model_name += f"_{model_names_dict[model_name]}"
            # set the model to evaluation mode
            target_model.eval()
            target_predictions = target_model(adversarial_images)

            target_model_predic_labels = torch.argmax(
                target_predictions, dim=1)

            # Count transfers where the adversarial example was also successful on the source model
            successful_transfer = torch.sum(successful_on_source_mask & (
                (target_model_predic_labels == target_labels) if is_targeted else (target_model_predic_labels != true_labels)))

            self.transferability_data[model_name]['successful_transfer'] += successful_transfer.item()

    def get_results(self) -> dict:
        results = {}
        for model_name, data in self.transferability_data.items():
            rate = (data['successful_transfer'] /
                    self.total_successful_on_source) if self.total_successful_on_source > 0 else 0
            results[model_name] = rate
        return results
