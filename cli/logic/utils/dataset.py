"""
CLI command functions related to datasets.
"""

import click

from advsecurenet.datasets.HuggingFace import HuggingFaceDataset
from advsecurenet.datasets import DatasetFactory
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.shared.types.configs.preprocess_config import HuggingFaceDatasetConfig
from cli.shared.types.utils.dataset import HuggingFaceDatasetCliConfigType


def cli_huggingface_dataset(dataset_id: str, subset: str, split: str, revision: str, trust_remote_code: bool):
    """
    Load and inspect a Hugging Face dataset.

    Args:
        dataset_id (str): Hugging Face dataset ID or URL.
        subset (str, optional): Subset of the dataset to use.
        split (str, optional): Split of the dataset to use.
        revision (str, optional): Specific dataset version to use.
        trust_remote_code (bool): Whether to trust remote code when loading the dataset.

    Raises:
        ValueError: If the dataset ID is not provided or is invalid.
    """
    if not dataset_id:
        raise click.ClickException("Dataset ID must be provided!")

    try:
        # Check if dataset_id is a URL and extract the dataset ID if it is
        if HuggingFaceDataset.is_huggingface_url(dataset_id):
            extracted_id = HuggingFaceDataset.extract_dataset_id_from_url(dataset_id)
            if extracted_id:
                click.echo(f"Detected Hugging Face URL. Using dataset ID: {extracted_id}")
                dataset_id_to_use = extracted_id
            else:
                raise click.ClickException(f"Could not extract dataset ID from URL: {dataset_id}")
        else:
            dataset_id_to_use = dataset_id

        # Create HuggingFaceDatasetConfig
        huggingface_config = HuggingFaceDatasetConfig(
            dataset_id=dataset_id_to_use,
            subset=subset,
            split=split,
            revision=revision,
            trust_remote_code=trust_remote_code
        )

        # Create dataset CLI config
        config = HuggingFaceDatasetCliConfigType(
            dataset_name="HUGGINGFACE",
            num_classes=2,  # Default, will be updated if possible
            dataset_id=dataset_id_to_use,
            subset=subset,
            split=split,
            revision=revision,
            trust_remote_code=trust_remote_code,
            huggingface_config=huggingface_config
        )

        click.echo(f"Loading Hugging Face dataset: {dataset_id_to_use}")
        if subset:
            click.echo(f"Subset: {subset}")
        if split:
            click.echo(f"Split: {split}")
        else:
            click.echo("Loading both train and test splits")

        # Create dataset object
        dataset_obj = DatasetFactory.create_dataset(
            dataset_type=DatasetType.HUGGINGFACE,
            huggingface_config=huggingface_config
        )

        # Display dataset information
        click.secho(f"Successfully loaded Hugging Face dataset: {dataset_id_to_use}", bold=True, fg="green")
        click.echo(f"Number of classes: {dataset_obj.num_classes}")
        click.echo(f"Number of input channels: {dataset_obj.num_input_channels}")
        if hasattr(dataset_obj, "input_size"):
            click.echo(f"Input size: {dataset_obj.input_size}")

        # Load a sample of the dataset to display information
        try:
            # Try to load train split
            train_split_to_use = split if split else "train"
            if not split:
                dataset_obj._split = train_split_to_use
            train_dataset = dataset_obj.load_dataset(train=True)
            click.secho(f"Train dataset loaded successfully", fg="green")
            click.echo(f"Train dataset size: {len(train_dataset)}")
            
            # Display a sample
            if len(train_dataset) > 0:
                click.echo("Sample data structure:")
                sample = train_dataset[0]
                if isinstance(sample, tuple) and len(sample) == 2:
                    data, label = sample
                    if hasattr(data, "shape"):
                        click.echo(f"Data shape: {data.shape}")
                    click.echo(f"Label: {label}")
        except Exception as e:
            click.secho(f"Warning: Could not load train dataset: {str(e)}", fg="yellow")

        try:
            # Try to load test split
            test_split_to_use = split if split else "test"
            if not split:
                dataset_obj._split = test_split_to_use
            test_dataset = dataset_obj.load_dataset(train=False)
            click.secho(f"Test dataset loaded successfully", fg="green")
            click.echo(f"Test dataset size: {len(test_dataset)}")
        except Exception as e:
            click.secho(f"Warning: Could not load test dataset: {str(e)}", fg="yellow")

    except Exception as e:
        raise click.ClickException(f"Error loading Hugging Face dataset: {str(e)}")
