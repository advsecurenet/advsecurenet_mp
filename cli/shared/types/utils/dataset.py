from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto

from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig

@dataclass
class BaseDatasetCliConfigType:
    dataset_name: str
    num_classes: int
    preprocessing: Optional[PreprocessConfig] = None

    
@dataclass
class UserSplitConfig:
    """
    User-provided configuration to override global settings for a specific split.
    All fields are optional. In a YAML file, this would be a dictionary under a
    logical split name (e.g., 'train', 'test').
    """
    #source: Optional[str] = None
    identifier: Optional[str] = None
    split_name: Optional[str] = None
    preprocessing: Optional[PreprocessConfig] = None
    dataset_arguments: Optional[Dict[str, Any]] = None

@dataclass
class ResolvedSplitConfig:
    """Internal, fully-resolved configuration for a single dataset split."""
    #source: Optional[str] = None
    identifier: str
    split: str
    preprocessing: Optional[PreprocessConfig]
    kwargs: Dict[str, Any]
    num_classes: int

@dataclass
class DatasetFinalType:
    dataset_name: str
    splits: dict[splits: str, ResolvedSplitConfig] = None

@dataclass
class AttacksDatasetCliConfigType(DatasetFinalType):
    """
    This dataclass is used to store the configuration of the dataset CLI used for attacks. It extends the DatasetCliConfigType. In addition to the attributes of the DatasetCliConfigType, it has the following attributes:

    Attributes:
        dataset_part (Optional[str]): The part of the dataset to be used for the attack. it can be train, test or all. This is valid if the dataset paths are not provided.
        random_sample_size (Optional[int]): The size of the random sample to be taken from the dataset.

    """

    dataset_part: Optional[str] = "test"
    random_sample_size: Optional[int] = None


@dataclass()
class CreateDatasetCliConfigType():
    """
    This dataclass is used to store the configuration of the dataset CLI.
    It extends the DatasetCliConfigType with Hugging Face specific attributes.

    Attributes:
        (Optinal) dataset_config (dict): The configuration for the Hugging Face dataset. Takes all the arguments of the datasets.load_dataset function
    """

    #source: Optional[str] = None
    dataset_name: str = None
    num_classes: int = 10

    identifier: Optional[str] = None
    splits: Optional[List] = None
    preprocessing: Optional[PreprocessConfig] = None
    dataset_arguments: Optional[Dict[str, Any]] = None

    split_config: Optional[Dict[str, UserSplitConfig]] = None



class IdentifierSource(Enum):
    DATASET_IDENTIFIER = auto()
    DATASET_NAME = auto()

@staticmethod
def determine_identifier_and_soruce(config: CreateDatasetCliConfigType):
    if config.identifier:
        identifier = config.identifier
        source = IdentifierSource.DATASET_IDENTIFIER
    else:
        identifier = config.dataset_name
        source = IdentifierSource.DATASET_NAME
    return identifier, source



"""@dataclass
class DatasetCliConfigType(BaseDatasetCliConfigType):
    """"""
    This dataclass is used to store the configuration of the dataset CLI.
    """"""
    train_dataset_path: Optional[str] = None
    test_dataset_path: Optional[str] = None
    """

"""@dataclass()
class HuggingFaceDatasetInputCliConfigType(BaseDatasetCliConfigType):
    """"""
    This dataclass is used to store the configuration of the Hugging Face dataset CLI.
    It extends the DatasetCliConfigType with Hugging Face specific attributes.

    Attributes:
        (Optinal) dataset_config (dict): The configuration for the Hugging Face dataset. Takes all the arguments of the datasets.load_dataset function
    """"""

    dataset_config: Optional[dict] = None

@dataclass(kw_only=True)
class HuggingFaceDatasetResolvedCliConfigType(HuggingFaceDatasetInputCliConfigType):
    dataset_id: str
    """
