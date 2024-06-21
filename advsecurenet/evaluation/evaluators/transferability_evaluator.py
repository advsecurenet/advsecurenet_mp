from typing import List, Optional

import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel


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
        model_names_count = {}

        for model in self.target_models:
            model_name = self._get_model_name(model, model_names_count)
            self.transferability_data[model_name] = {
                'successful_transfer': 0, 'successful_on_source': 0}

        self.total_successful_on_source = 0

    def __enter__(self):
        self.reset()
        return self

    def update(self, model: BaseModel, original_images: torch.Tensor, true_labels: torch.Tensor, adversarial_images: torch.Tensor, is_targeted: bool = False, target_labels: Optional[torch.Tensor] = None) -> None:
        """
        Update the transferability evaluator with new data.

        Args:
            model (BaseModel): The model to evaluate transferability on.
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels of the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool, optional): Whether the attack is targeted or not. Defaults to False.
            target_labels (torch.Tensor, optional): The target labels for targeted attacks. Defaults to None.

        Returns:
            None
        """
        device = next(model.parameters()).device
        original_images, adversarial_images, true_labels, target_labels = self._prepare_tensors(
            device, original_images, adversarial_images, true_labels, is_targeted, target_labels)

        model.eval()
        successful_on_source_mask, filtered_adversarial_images, filtered_true_labels, filtered_target_labels = self._get_successful_on_source_mask(
            model, original_images, true_labels, adversarial_images, is_targeted, target_labels)
        if successful_on_source_mask.numel() == 0:
            return  # No successful adversarial examples to evaluate

        self._evaluate_transferability(filtered_adversarial_images, filtered_true_labels,
                                       successful_on_source_mask, is_targeted, filtered_target_labels)

    def _get_successful_on_source_mask(self,
                                       model: torch.nn.Module,
                                       original_images: torch.Tensor,
                                       true_labels: torch.Tensor,
                                       adversarial_images: torch.Tensor,
                                       is_targeted: bool,
                                       target_labels: Optional[torch.Tensor]
                                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Identifies the successful adversarial examples on the source model. This method filters the data based on the correct initial predictions. This is important to avoid evaluating the transferability of unsuccessful adversarial examples.

        Args:
            model (torch.nn.Module): The source model used to generate adversarial examples.
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels of the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor]): The target labels for the targeted attack.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: The mask identifying successful adversarial examples on the source model, the filtered adversarial images, the filtered true labels, and the filtered target labels if the attack is targeted.

        """
        device = next(model.parameters()).device

        original_images = original_images.to(device)
        true_labels = true_labels.to(device)
        adversarial_images = adversarial_images.to(device)
        if is_targeted:
            if target_labels is None:
                raise ValueError(
                    "Target labels must be provided for targeted attacks.")
            target_labels = target_labels.to(device)

        initial_predictions = model(original_images)
        initial_labels = torch.argmax(initial_predictions, dim=1)

        # Mask to identify correct initial predictions
        correct_initial_predictions_mask = initial_labels == true_labels
        total_correct_initial = correct_initial_predictions_mask.sum().item()

        if total_correct_initial == 0:
            return torch.tensor([], device=device), adversarial_images, true_labels, target_labels if is_targeted else None

        # Filter the data based on the correct initial predictions
        filtered_adversarial_images = adversarial_images[correct_initial_predictions_mask]
        filtered_true_labels = true_labels[correct_initial_predictions_mask]
        filtered_target_labels = target_labels[correct_initial_predictions_mask] if is_targeted else None

        source_predictions = model(filtered_adversarial_images)
        source_labels = torch.argmax(source_predictions, dim=1)

        successful_on_source_mask = (source_labels == filtered_target_labels) if is_targeted else (
            source_labels != filtered_true_labels)
        self.total_successful_on_source += successful_on_source_mask.sum().item()

        return successful_on_source_mask, filtered_adversarial_images, filtered_true_labels, filtered_target_labels

    def _prepare_tensors(self,
                         device: torch.device,
                         original_images: torch.Tensor,
                         adversarial_images: torch.Tensor,
                         true_labels: torch.Tensor,
                         is_targeted: bool,
                         target_labels: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares the tensors for evaluation.

        Args:
            device (torch.device): The device to move the tensors to.
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.
            true_labels (torch.Tensor): The true labels of the original images.
            is_targeted (bool): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor]): The target labels for the targeted attack.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: The prepared tensors.
        """

        original_images = original_images.to(device)
        adversarial_images = adversarial_images.to(device)
        true_labels = true_labels.to(device)

        if is_targeted:
            if target_labels is None:
                raise ValueError(
                    "Target labels must be provided for targeted attacks.")
            target_labels = target_labels.to(device)

        return original_images, adversarial_images, true_labels, target_labels

    def _evaluate_transferability(self,
                                  adversarial_images: torch.Tensor,
                                  true_labels: torch.Tensor,
                                  successful_on_source_mask: torch.Tensor,
                                  is_targeted: bool,
                                  target_labels: Optional[torch.Tensor]) -> None:
        """
        Evaluates the transferability of adversarial examples to target models.

        Args:
            adversarial_images (torch.Tensor): The adversarial images.
            true_labels (torch.Tensor): The true labels of the original images.
            successful_on_source_mask (torch.Tensor): Mask identifying successful adversarial examples on the source model.
            is_targeted (bool): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor]): The target labels for the targeted attack.
        """
        model_names_count = {}

        for target_model in self.target_models:
            model_name = self._get_model_name(target_model, model_names_count)
            successful_transfer = self._evaluate_model_transferability(
                target_model, adversarial_images, true_labels, successful_on_source_mask, is_targeted, target_labels)

            self.transferability_data[model_name]['successful_transfer'] += successful_transfer.item()
            self.transferability_data[model_name]['successful_on_source'] += successful_on_source_mask.sum().item()

            # Move tensors to CPU to avoid memory leaks and GPU memory overflow
            self._move_tensors_to_cpu(
                adversarial_images, true_labels, successful_on_source_mask, is_targeted, target_labels)

    def _get_model_name(self, target_model, model_names_count):
        """
        Generates a unique name for the target model.

        Args:
            target_model (torch.nn.Module): The target model.
            model_names_count (dict): Dictionary to keep track of the number of models of the same type.

        Returns:
            str: The unique name for the target model.
        """
        model_name = target_model._model_name
        model_names_count[model_name] = model_names_count.get(
            model_name, 0) + 1
        if model_names_count[model_name] == 1:
            return model_name
        else:
            return f"{model_name}_{model_names_count[model_name]}"

    def _evaluate_model_transferability(self,
                                        target_model: torch.nn.Module,
                                        adversarial_images: torch.Tensor,
                                        true_labels: torch.Tensor,
                                        successful_on_source_mask: torch.Tensor,
                                        is_targeted: bool,
                                        target_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Evaluates the transferability of adversarial examples to a target model.

        Args:
            target_model (torch.nn.Module): The target model to evaluate.
            adversarial_images (torch.Tensor): The adversarial images.
            true_labels (torch.Tensor): The true labels of the original images.
            successful_on_source_mask (torch.Tensor): Mask identifying successful adversarial examples on the source model.
            is_targeted (bool): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor]): The target labels for the targeted attack.

        Returns:
            torch.Tensor: The number of successful transfers to the target model.
        """
        target_model.eval()
        device = next(target_model.parameters()).device

        adv_images, labels, mask = map(lambda x: x.to(device),
                                       [adversarial_images, true_labels, successful_on_source_mask])
        if is_targeted:
            target_labels = target_labels.to(device)

        target_predictions = target_model(adv_images)
        target_labels_pred = torch.argmax(target_predictions, dim=1)

        successful_transfer = torch.sum(mask & (
            (target_labels_pred == target_labels) if is_targeted else (target_labels_pred != labels)))

        return successful_transfer

    def _move_tensors_to_cpu(self,
                             adversarial_images: torch.Tensor,
                             true_labels: torch.Tensor,
                             successful_on_source_mask: torch.Tensor,
                             is_targeted: bool,
                             target_labels: Optional[torch.Tensor]) -> None:
        """
        Moves tensors to CPU to avoid memory leaks and GPU memory overflow.

        Args:
            adversarial_images (torch.Tensor): The adversarial images.
            true_labels (torch.Tensor): The true labels of the original images.
            successful_on_source_mask (torch.Tensor): Mask identifying successful adversarial examples on the source model.
            is_targeted (bool): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor]): The target labels for the targeted attack.

        """

        adversarial_images = adversarial_images.cpu()
        true_labels = true_labels.cpu()
        successful_on_source_mask = successful_on_source_mask.cpu()
        if is_targeted and target_labels is not None:
            target_labels = target_labels.cpu()

    def get_results(self) -> dict:
        """
        Calculates the transferability results for the streaming session and returns them.

        Returns:
            dict: Transferability results for each target model.
        """
        results = {}
        for model_name, data in self.transferability_data.items():
            rate = (data['successful_transfer'] /
                    self.total_successful_on_source) if self.total_successful_on_source > 0 else 0
            results[model_name] = rate
        return results
