from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Optional, Tuple
import inspect

import torch
from torch import nn
from torchvision.models.feature_extraction import get_graph_node_names


def check_model_loaded(func):
    """
    Wrapper function to check if the model is loaded before calling the decorated function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return func(self, *args, **kwargs)

    return wrapper


class BaseModel(ABC, nn.Module):
    """
    Abstract class for models.

    Attributes:
        architecture (Dict[str, Any]): Any argument for the initialisaion of the Model
        pretrained (bool): Whether to load the pretrained weights or not.
        model_name (str): Model Name
    """

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """
        Abstract method to load the model. This method should be implemented
        in derived classes (e.g., StandardModel, CustomModel).
        """

    @check_model_loaded
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default forward pass. Assumes self.model returns a logits tensor directly.
        Subclasses handling complex output objects should override this method.

        Args:
            x (torch.Tensor): The primary input tensor to the model.
            *args: Additional positional arguments to pass to the underlying model's forward method.
            **kwargs: Additional keyword arguments to pass to the underlying model's forward method.

        Returns:
            torch.Tensor: The output logits tensor from the model.

        Raises:
            TypeError: If the underlying model's output is not a Tensor and this method hasn't been overridden by a subclass.
        """
        return self.model(x)

    @check_model_loaded
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the class of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The predicted class index.
                - The probability of the predicted class.
        """
        logits = self.forward(x)
        probabilities = nn.Softmax(dim=1)(logits)
        predicted_classes = logits.argmax(dim=1)
        max_probabilities = probabilities.max(dim=1)[0]
        return predicted_classes, max_probabilities

    @check_model_loaded
    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    @abstractmethod
    def models(self):
        """
        Return a list of available models.
        """
        pass

    @check_model_loaded
    def get_layer_names(self) -> List[str]:
        """
        Return a list of layer names in the model.
        """
        try:
            # Try the torch.fx-based method
            _, eval_nodes = get_graph_node_names(self.model)
            return eval_nodes
        except torch.fx.proxy.TraceError:
            # Fallback: use named_modules to get layer names
            return [name for name, _ in self.model.named_modules() if name]

    @check_model_loaded
    def get_layer(self, layer_name: str) -> nn.Module:
        """
        Retrieve a specific layer module based on its name.

        Examples:
            >>> model = StandardModel(model_name='resnet18', architecture{\"num_classes\": 10})
            >>> model.get_layer('layer1.0.conv1')
        """
        return dict(self.model.named_modules()).get(layer_name, None)

    @check_model_loaded
    def set_layer(self, layer_name: str, new_layer: nn.Module):
        """
        Replace a specific layer module based on its name with a new module.

        Examples:
            >>> model = StandardModel(model_name='resnet18', architecture{\"num_classes\": 10})
            >>> model.set_layer('layer1.0.conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        """

        # Obtain the parent module and the attribute name of the layer
        parent, name = self._get_parent_module_and_name(layer_name)
        # Set the new layer
        setattr(parent, name, new_layer)

    @check_model_loaded
    def add_layer(
        self, new_layer: nn.Module, position: int = -1, inplace: bool = True
    ) -> Optional[nn.Module]:
        """
        Inserts a new layer into the model at the specified position.

        Args:
            new_layer (nn.Module): The new layer to be added.
            position (int): The position at which the new layer should be added. By default, it is added at the end.
            inplace (bool): Whether to add the layer in-place or not. If set to False, a new model is created.

        Returns:
            Optional[nn.Module]: The new model if inplace is set to False.

        """

        if not isinstance(self.model, nn.Sequential):
            # convert the model to a Sequential model
            self.model = nn.Sequential(self.model)

        layers = list(self.model.children())

        # Check if the position is out of bounds
        if position < -1 or position > len(layers):
            raise ValueError(f"Invalid position: {position}")

        if position == -1 or position == len(layers):
            layers.append(new_layer)

        else:
            layers.insert(position, new_layer)

        if inplace:
            self.model = nn.Sequential(*layers)
        else:
            return nn.Sequential(*layers)

    @check_model_loaded
    def _get_parent_module_and_name(self, layer_name: str) -> Tuple[nn.Module, str]:
        """
        Helper method to get the parent module and the attribute name of a layer.
        """

        if "." in layer_name:
            parent_name, child_name = layer_name.rsplit(".", 1)
            parent = dict(self.model.named_modules()).get(parent_name, self.model)
        else:
            parent = self.model
            child_name = layer_name
        return parent, child_name
    
    @check_model_loaded
    def infer_num_classes(self) -> Optional[int]:
        """
        Infers the number of output classes based on the model's architecture.
    
        Returns:
            Optional[int]: The inferred number of classes, or None if it cannot be inferred.
        """
        # For Hugging Face models, try extracting from the configuration.
        if hasattr(self.model, "config") and hasattr(self.model.config, "num_labels"):
            return self.model.config.num_labels

        # For models with a classifier attribute.
        if hasattr(self.model, "classifier"):
            # If classifier is a single linear layer.
            if hasattr(self.model.classifier, "out_features"):
                return self.model.classifier.out_features
            # If classifier is a sequential container, try its last module.
            elif isinstance(self.model.classifier, nn.Sequential):
                last_layer = list(self.model.classifier.children())[-1]
                if hasattr(last_layer, "out_features"):
                    return last_layer.out_features

        # For models with an fc attribute (common in ResNet-like architectures).
        if hasattr(self.model, "fc") and hasattr(self.model.fc, "out_features"):
            return self.model.fc.out_features

        return None
    
    def _filter_architecture_params(self, model_class: type, architecture: dict, model_name: str) -> nn.Module:
        """
        Filter the architectures parameters that don't match with the model_class parameters
        to match the model's __init__ signature.

        Args:
            model_class (type): The model class to instantiate.
            architecture (dict): The dictionary of parameters to potentially pass to the model's __init__.
            model_name (str): The name of the model (used for error messages).

        Returns:
            nn.Module: The instantiated model.

        Raises:
            RuntimeError: If inspection or instantiation fails.
        """
        try:
            # Get the signature of the model's __init__ method
            # Handle cases where __init__ might be inherited or not explicitly defined
            init_method = getattr(model_class, '__init__', object.__init__)
            sig = inspect.signature(init_method)
            accepted_params = set(sig.parameters.keys())
            # Remove 'self' if present, as it's implicitly passed
            accepted_params.discard('self')

            # Filter the architecture dictionary to include only accepted parameters
            filtered_architecture = {
                k: v for k, v in architecture.items() if k in accepted_params
            }
            return filtered_architecture

        except Exception as e:
             # Catch potential errors during inspection or instantiation
             raise RuntimeError(f"Failed to inspect or instantiate model {model_name} with provided architecture: {e}") from e