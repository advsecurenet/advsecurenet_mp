import click


@click.group()
def models():
    """
    Command to list available models.
    """


@models.command()
@click.option(
    "-m",
    "--model-type",
    default="all",
    help="The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.",
)
def list(model_type: str):
    """Command to list available models.

    Args:

        model_type (str, optional): The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.

    Raises:
        ValueError: If the model_type is not supported.
    """
    from cli.logic.utils.model import cli_models

    cli_models(model_type)


@models.command()
@click.option(
    "-m",
    "--model-name",
    default=None,
    help='Name of the model to inspect (e.g. "resnet18").',
)
@click.option(
    "-n",
    "--normalization",
    is_flag=True,
    type=click.BOOL,
    default=False,
    help="Whether to include normalization layer in the model summary.",
)
def layers(model_name: str, normalization: bool):
    """Command to list the layers of a model.

    Args:

        model_name (str): The name of the model (e.g. "resnet18").
        normalization (bool): Whether to include normalization layer in the model summary.

    Raises:
        ValueError: If the model name is not provided.
    """
    from cli.logic.utils.model import cli_model_layers

    cli_model_layers(model_name, normalization)

@models.command()
@click.option(
    "-m",
    "--model-url",
    required=True,
    help="Hugging Face model URL (e.g., 'https://huggingface.co/bert-base-uncased') .",
)
@click.option(
    "-c",
    "--num-classes",
    type=int,
    default=None,
    help="Number of classes for the model. If not provided, will use the default from the model.",
)
@click.option(
    "-p",
    "--pretrained",
    is_flag=True,
    type=click.BOOL,
    default=True,
    help="Whether to use pretrained weights. Default is True.",
)
@click.option(
    "-r",
    "--revision",
    default=None,
    help="Specific model version to use (e.g., 'main', 'v1.0'). If not provided, will use the default.",
)
@click.option(
    "-t",
    "--trust-remote-code",
    is_flag=True,
    type=click.BOOL,
    default=False,
    help="Whether to trust remote code when loading the model. Default is False.",
)
def huggingface(model_url: str, pretrained: bool, revision: str, trust_remote_code: bool):
    """Command to load and inspect a Hugging Face model.

    Args:
        model_url (str): Hugging Face model ID or URL.
        pretrained (bool): Whether to use pretrained weights.
        revision (str, optional): Specific model version to use.
        trust_remote_code (bool): Whether to trust remote code when loading the model.

    Raises:
        ValueError: If the model ID is not provided or is invalid.
    """
    from cli.logic.utils.model import cli_huggingface_model

    cli_huggingface_model(model_url, pretrained, revision, trust_remote_code)