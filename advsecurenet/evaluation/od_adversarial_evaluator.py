from typing import Optional, List, Dict, Any
import torch
from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.models.base_model import BaseModel


class ObjectDetectorAdversarialEvaluator(AdversarialEvaluator):
    """
    Evaluator for object detection attacks using mean Average Precision (mAP).

    Args:
        evaluators (Optional[list[str]], optional): List of evaluators to use. If None, mAP evaluator will be used. Defaults to None.
        **kwargs: Arbitrary keyword arguments for the evaluators.
    """

    def __init__(
        self,
        evaluators: Optional[list[str]] = None,
        dataset_name: Optional[str] = "coco",
        **kwargs
    ):
        if evaluators is None:
            evaluators = ["mean_average_precision"]
        super().__init__(evaluators=evaluators, **kwargs)
        self.save_dataset_name(dataset_name)

    def save_dataset_name(self, dataset_name: str):
        if "mean_average_precision" in self.selected_evaluators:
            self.selected_evaluators["mean_average_precision"].dataset_name = (
                dataset_name
            )

    def update(
        self,
        model: BaseModel,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        targets: List[Dict[str, Any]],
    ) -> None:
        """
        Updates the evaluator with new data for streaming mode (object detection).
        Args:
            model (BaseModel): The object detection model to evaluate.
            images (torch.Tensor): A batch of images to run predictions on.
            targets (List[Dict[str, Any]]): A list of ground truth dictionaries, each with 'boxes' and 'labels'.
        """
        if "mean_average_precision" in self.selected_evaluators:
            self.evaluators["mean_average_precision"].update(
                model, original_images, adversarial_images, targets
            )
