import click

@click.group()
def datasets():
    """
    Command to manage datasets.
    """

@datasets.command()
@click.option(
    "-d",
    "--dataset-id",
    required=True,
    help="Hugging Face dataset ID or URL (e.g., 'mnist' or 'https://huggingface.co/datasets/mnist') .",
)
@click.option(
    "-s",
    "--subset",
    default=None,
    help="Subset of the dataset to use (e.g., 'plain_text' for text datasets).",
)
@click.option(
    "-p",
    "--split",
    default=None,
    help="Split of the dataset to use (e.g., 'train', 'test'). If not provided, will load both train and test splits.",
)
@click.option(
    "-r",
    "--revision",
    default=None,
    help="Specific dataset version to use (e.g., 'main', 'v1.0'). If not provided, will use the default.",
)
@click.option(
    "-t",
    "--trust-remote-code",
    is_flag=True,
    type=click.BOOL,
    default=False,
    help="Whether to trust remote code when loading the dataset. Default is False.",
)
def huggingface(dataset_id: str, subset: str, split: str, revision: str, trust_remote_code: bool):
    """Command to load and inspect a Hugging Face dataset.

    Args:
        dataset_id (str): Hugging Face dataset ID or URL.
        subset (str, optional): Subset of the dataset to use.
        split (str, optional): Split of the dataset to use.
        revision (str, optional): Specific dataset version to use.
        trust_remote_code (bool): Whether to trust remote code when loading the dataset.

    Raises:
        ValueError: If the dataset ID is not provided or is invalid.
    """
    from cli.logic.utils.dataset import cli_huggingface_dataset

    cli_huggingface_dataset(dataset_id, subset, split, revision, trust_remote_code)
