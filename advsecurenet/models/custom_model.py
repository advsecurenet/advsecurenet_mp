import importlib
import os

from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.model_config import CustomModelConfig
from advsecurenet.utils.kwargs_utils import filter_kwargs_for_callable


class CustomModel(BaseModel):
    """
    This class is used to load a custom model. It is a subclass of BaseModel.
    """

    def __init__(self, config: CustomModelConfig):
        self._custom_models_path = config.custom_models_path
        self._model_name = config.model_name
        self._architecture = config.architecture

        # Initialize the BaseModel
        super().__init__()

    def load_model(self):
        """
        Load the custom model. This method is called by the BaseModel constructor.
        It dynamically imports the custom model based on its name. That is, it assumes that the model class inside the custom model file has the same name as the file
        i.e., for 'CustomMnistModel.py', there should be a class 'CustomMnistModel'.

        Raises
        ------
        ValueError
            If the model class is not found in the custom model file.
        """

        # Dynamically import the custom model based on its name
        custom_module_name = (
            f"advsecurenet.models.{self._custom_models_path}.{self._model_name}"
        )
        custom_module = importlib.import_module(custom_module_name)

        # Assume the model class inside the custom model file has the same name as the file
        if not hasattr(custom_module, self._model_name):
            raise ValueError(
                f"Model class {self._model_name} not found in module {custom_module_name}"
            )

        model_class = getattr(custom_module, self._model_name)

        filtered_architecture = filter_kwargs_for_callable(model_class, self._architecture)
        self.model = model_class(**filtered_architecture)

        # Perform necessary modifications after model load
        self.modify_model()

    def modify_model(self):
        # Usually, for custom models, you might not need modifications after loading.
        # But if you do, you can specify them here.
        pass

    @staticmethod
    def models():
        """
        Returns a list of available custom models from the CustomModels directory.

        Returns
        -------
        List[str]
            A list of available custom models.
        """

        # Path to the CustomModels directory
        custom_models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "CustomModels"
        )

        # List all python files in the CustomModels directory
        all_files = [
            f
            for f in os.listdir(custom_models_dir)
            if os.path.isfile(os.path.join(custom_models_dir, f)) and f.endswith(".py")
        ]

        # Extract the model names by stripping the '.py' from filenames
        model_names = [os.path.splitext(f)[0] for f in all_files]

        # remove __init__ from model names
        model_names = [
            model_name for model_name in model_names if model_name != "__init__"
        ]

        return model_names

    @staticmethod
    def available_weights(model_name: str) -> list[str]:
        """
        Not applicable for custom models.
        """
        raise NotImplementedError("This method is not applicable for custom models.")
