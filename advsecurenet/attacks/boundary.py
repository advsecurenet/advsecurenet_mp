import typing
from collections import deque
from typing import Optional

import click
import torch
import torch.nn as nn
from tqdm.auto import trange

import advsecurenet.shared.types.configs.attack_configs as AttackConfigs
from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel


class DecisionBoundary(AdversarialAttack):
    """
    Decision Boundary attack

    Paper: https://arxiv.org/abs/1712.04248
    """

    def __init__(self, config: AttackConfigs.DecisionBoundaryAttackConfig) -> None:
        self.initial_delta = config.initial_delta
        self.initial_epsilon = config.initial_epsilon
        self.max_delta_trials = config.max_delta_trials
        self.max_epsilon_trials = config.max_epsilon_trials
        self.max_iterations = config.max_iterations
        self.max_initialization_trials = config.max_initialization_trials
        self.step_adapt = config.step_adapt
        self.targeted = config.targeted
        self.verbose = config.verbose
        self.early_stopping = config.early_stopping
        self.early_stopping_threshold = config.early_stopping_threshold
        self.early_stopping_patience = config.early_stopping_patience
        super().__init__(config)

    @typing.no_type_check
    def attack(self, model: BaseModel, original_images: torch.Tensor, true_labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        if self.targeted and target_labels is None:
            raise ValueError(
                "Target labels must be provided for a targeted attack")

        if self.targeted and target_labels is not None:
            target_labels = self.device_manager.to_device(target_labels)

        # model.eval()

        adv_images = self._initialize(
            model, original_images, true_labels, target_labels)

        adv_images = self.device_manager.to_device(adv_images)
        best_adv_images, best_distances = self._initialize_best_images(
            adv_images, original_images)

        delta = self.initial_delta
        epsilon = self.initial_epsilon

        for iteration in trange(self.max_iterations, desc="Boundary Attack", colour="blue", disable=not self.verbose):
            adv_images, delta = self._perturb_orthogonal(
                model, original_images, true_labels, target_labels, adv_images, delta)
            adv_images, epsilon = self._perturb_forward(
                model, original_images, true_labels, target_labels, adv_images, epsilon)

        best_adv_images, best_distances = self._update_best_images(
            adv_images, original_images, best_adv_images, best_distances)

        return best_adv_images

    def _initialize(self, model, original_images, true_labels, target_labels=None):
        """
        Initializes the perturbed images for the Decision Boundary attack. The initialization is done by randomly and trying to find an adversarial example in a given number of iterations.
        If the attack is targeted, initialization tries to find an adversarial example that is classified as the target label. If the attack is untargeted, initialization tries to find an adversarial example that is misclassified (i.e. not classified as the true label).
        """

        # Ensure original_images, true_labels, and target_labels are all tensors on the same device
        original_images = self.device_manager.to_device(original_images)
        true_labels = self.device_manager.to_device(true_labels)

        if target_labels is not None:
            target_labels = self.device_manager.to_device(target_labels)

        # Create a mask for images that need to be updated, initially all are True
        update_mask = torch.ones(original_images.size(
            0), dtype=torch.bool, device=original_images.device)

        perturbed_images = original_images.clone()

        for _ in trange(self.max_initialization_trials, desc="Initializing", colour="yellow", disable=not self.verbose):
            # Randomly initialize the perturbed images for those that need to be updated
            random_imgs = torch.rand_like(original_images)
            perturbed_images[update_mask] = random_imgs[update_mask]

            # Forward pass with the perturbed images
            outputs = model(perturbed_images)
            preds = outputs.argmax(dim=1)

            # Update the mask based on whether the attack is targeted or untargeted
            if self.targeted:
                assert target_labels is not None, "Target labels must be provided for a targeted attack"
                # Ensure this is a boolean tensor
                update_mask = ~(preds == target_labels)
            else:
                update_mask = preds == true_labels  # Ensure this is a boolean tensor

        return perturbed_images

    def _orthogonal_perturb(self, delta, current_samples, original_samples):
        batch_size, channels, height, width = current_samples.shape
        # Generate perturbation randomly for a batch of images
        perturb = torch.randn_like(current_samples)

        # Rescale the perturbation
        perturb_norm = torch.norm(perturb.view(
            batch_size, -1), dim=1, keepdim=True)
        perturb = perturb / perturb_norm.view(batch_size, 1, 1, 1)

        diff = original_samples - current_samples
        diff_norm = torch.norm(diff.view(batch_size, -1), dim=1, keepdim=True)
        perturb = perturb * (delta * diff_norm.view(batch_size, 1, 1, 1))

        # Project the perturbation onto the sphere
        direction = diff / diff_norm.view(batch_size, 1, 1, 1)
        perturb_flat = perturb.view(batch_size, -1)
        direction_flat = direction.view(batch_size, -1)

        # Remove component in the direction of (original - current)
        adjustment = torch.bmm(perturb_flat.unsqueeze(
            1), direction_flat.unsqueeze(2)).squeeze(2)
        perturb_flat -= adjustment * direction_flat
        perturb = perturb_flat.view_as(perturb)

        # Calculate the final perturbed image
        delta = torch.tensor(delta, dtype=torch.float32,
                             device=current_samples.device)
        hypotenuse = torch.sqrt(1 + delta ** 2)
        perturb = ((1 - hypotenuse) * diff + perturb) / hypotenuse

        return perturb

    def _forward_perturb(self, epsilon, adv_images, original_images):
        """
        Generates a perturbation in the direction of the original image.

        Args:
            epsilon (float): The epsilon value to use for the attack.
            adv_images (torch.tensor): The adversarial images. Expected shape is (batch_size, channels, height, width).
            original_images (torch.tensor): The original images. Expected shape is (batch_size, channels, height, width).

        Returns:

            torch.tensor: The perturbation tensor.
        """
        # Calculate the direction vector from the adversarial images towards the original images
        direction = original_images - adv_images
        # Calculate the norm of the direction (batch-wise)
        norm = torch.norm(direction.view(direction.size(0), -1),
                          p=2, dim=1).view(-1, 1, 1, 1)
        # Avoid division by zero
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        # Calculate the perturbation
        perturbation = epsilon * direction / norm

        return perturbation

    def _initialize_best_images(self, adv_images: torch.Tensor, original_images: torch.Tensor):
        best_adv_images = adv_images.clone()
        best_distances = torch.full((adv_images.size(0),), float(
            'inf'), device=original_images.device)
        return best_adv_images, best_distances

    def _perturb_orthogonal(self, model: nn.Module, original_images: torch.Tensor, true_labels: torch.Tensor, target_labels: Optional[torch.Tensor], adv_images: torch.Tensor, delta: float) -> tuple:
        for _ in range(self.max_delta_trials):
            perturbation = self._orthogonal_perturb(
                delta, adv_images, original_images)
            trial_images = adv_images + perturbation
            trial_images.clamp_(min=0, max=1)

            outputs = model(trial_images)
            predictions = outputs.argmax(dim=1)
            success = self._evaluate_success(
                predictions, true_labels, target_labels)

            delta = self._adjust_delta(success, delta)
            adv_images = self._update_adv_images(
                success, adv_images, trial_images)

        return adv_images, delta

    def _perturb_forward(self, model: nn.Module, original_images: torch.Tensor, true_labels: torch.Tensor, target_labels: Optional[torch.Tensor], adv_images: torch.Tensor, epsilon: float) -> tuple:
        for _ in range(self.max_epsilon_trials):
            perturbation = self._forward_perturb(
                epsilon, adv_images, original_images)
            trial_images = adv_images + perturbation
            trial_images.clamp_(min=0, max=1)

            outputs = model(trial_images)
            predictions = outputs.argmax(dim=1)
            success = self._evaluate_success(
                predictions, true_labels, target_labels)

            epsilon = self._adjust_epsilon(success, epsilon)
            adv_images = self._update_adv_images(
                success, adv_images, trial_images)

        return adv_images, epsilon

    def _evaluate_success(self, predictions: torch.Tensor, true_labels: torch.Tensor, target_labels: Optional[torch.Tensor]) -> torch.Tensor:
        if self.targeted:
            return predictions == target_labels
        else:
            return predictions != true_labels

    def _adjust_delta(self, success: torch.Tensor, delta: float) -> float:
        success_rate = success.float().mean().item()
        if success_rate < 0.2:
            delta *= self.step_adapt
        elif success_rate > 0.5:
            delta /= self.step_adapt
        return delta

    def _adjust_epsilon(self, success: torch.Tensor, epsilon: float) -> float:
        success_rate = success.float().mean().item()
        if success_rate < 0.2:
            epsilon *= self.step_adapt
        elif success_rate > 0.5:
            epsilon /= self.step_adapt
        return epsilon

    def _update_adv_images(self, success: torch.Tensor, adv_images: torch.Tensor, trial_images: torch.Tensor) -> torch.Tensor:
        for idx in range(adv_images.size(0)):
            if success[idx]:
                adv_images[idx] = trial_images[idx]
        return adv_images

    def _update_best_images(self, adv_images: torch.Tensor, original_images: torch.Tensor, best_adv_images: torch.Tensor, best_distances: torch.Tensor) -> tuple:
        with torch.no_grad():
            distances = torch.norm(
                adv_images - original_images, p=2, dim=(1, 2, 3))
            improved = distances < best_distances
            best_adv_images[improved] = adv_images[improved]
            best_distances[improved] = distances[improved]
        return best_adv_images, best_distances

    def _check_early_stopping(self, best_distances: torch.Tensor, recent_improvements: deque, iteration: int) -> bool:
        if iteration >= self.early_stopping_patience:
            improvement = recent_improvements[0] - best_distances.mean().item()
            recent_improvements.append(best_distances.mean().item())
            if improvement < self.early_stopping_threshold:
                if self.verbose:
                    click.echo(click.style("Early stopping", fg="yellow"))
                return True
        else:
            recent_improvements.append(best_distances.mean().item())
        return False
