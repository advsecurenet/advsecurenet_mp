from dataclasses import dataclass
from typing import Optional

from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig

@dataclass
class BaseDatasetCliConfigType:
    dataset_name: str
    num_classes: int


@dataclass
class DatasetCliConfigType(BaseDatasetCliConfigType):
    """
    This dataclass is used to store the configuration of the dataset CLI.
    """
    train_dataset_path: Optional[str] = None
    test_dataset_path: Optional[str] = None
    download: Optional[bool] = True
    preprocessing: Optional[PreprocessConfig] = None


@dataclass
class AttacksDatasetCliConfigType(DatasetCliConfigType):
    """
    This dataclass is used to store the configuration of the dataset CLI used for attacks. It extends the DatasetCliConfigType. In addition to the attributes of the DatasetCliConfigType, it has the following attributes:

    Attributes:
        dataset_part (Optional[str]): The part of the dataset to be used for the attack. it can be train, test or all. This is valid if the dataset paths are not provided.
        random_sample_size (Optional[int]): The size of the random sample to be taken from the dataset.

    """

    dataset_part: Optional[str] = "test"
    random_sample_size: Optional[int] = None

@dataclass()
class HuggingFaceDatasetInputCliConfigType(BaseDatasetCliConfigType):
    """
    This dataclass is used to store the configuration of the Hugging Face dataset CLI.
    It extends the DatasetCliConfigType with Hugging Face specific attributes.

    Attributes:
        (Optinal) dataset_config (dict): The configuration for the Hugging Face dataset. Takes all the arguments of the datasets.load_dataset function
    """

    dataset_config: Optional[dict] = None

@dataclass(kw_only=True)
class HuggingFaceDatasetResolvedCliConfigType(HuggingFaceDatasetInputCliConfigType):
    dataset_id: str

@dataclass()
class CreateDatasetCliConfigType(DatasetCliConfigType, HuggingFaceDatasetInputCliConfigType):
    """
    This dataclass is used to store the configuration of the dataset CLI.
    It extends the DatasetCliConfigType with Hugging Face specific attributes.

    Attributes:
        (Optinal) dataset_config (dict): The configuration for the Hugging Face dataset. Takes all the arguments of the datasets.load_dataset function
    """

    dataset_identifier: str = None
