from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.commands.utils.models.commands import models


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.cli_models")
def test_list_command(mock_cli_models, runner):
    result = runner.invoke(models, ["list", "--model-type", "custom"])
    assert result.exit_code == 0
    mock_cli_models.assert_called_once_with("custom")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.cli_models")
def test_list_command_default(mock_cli_models, runner):
    result = runner.invoke(models, ["list"])
    assert result.exit_code == 0
    mock_cli_models.assert_called_once_with("all")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.cli_model_layers")
def test_layers_command(mock_cli_model_layers, runner):
    result = runner.invoke(
        models, ["layers", "--model-name", "resnet18", "--normalization"]
    )
    assert result.exit_code == 0
    mock_cli_model_layers.assert_called_once_with("resnet18", True)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.cli_model_layers")
def test_layers_command_no_normalization(mock_cli_model_layers, runner):
    result = runner.invoke(models, ["layers", "--model-name", "resnet18"])
    assert result.exit_code == 0
    mock_cli_model_layers.assert_called_once_with("resnet18", False)


@pytest.mark.cli
@pytest.mark.essential
def test_huggingface_command_missing_identifier(runner):
    # Test that it fails if the required model-identifier is not provided
    result = runner.invoke(models, ["huggingface"])
    assert result.exit_code != 0
    assert "Missing option '-i' / '--model-identifier'" in result.output


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.commands.utils.models.commands.cli_huggingface_model")
def test_huggingface_command_with_valid_identifier(mock_cli_huggingface_model, runner):
    # Test the successful execution path with valid parameters
    result = runner.invoke(models, ["huggingface", "-i", "bert-base-uncased"])
    assert result.exit_code == 0
    mock_cli_huggingface_model.assert_called_once_with(
        model_identifier="bert-base-uncased",
        pretrained=True,
        revision=None,
        trust_remote_code=False,
        model_class_name=None,
    )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.commands.utils.models.commands.cli_huggingface_model")
def test_huggingface_command_with_all_options(mock_cli_huggingface_model, runner):
    # Test with all optional parameters set
    result = runner.invoke(
        models,
        [
            "huggingface",
            "-i",
            "bert-base-uncased",
            "-r",
            "v1.0",
            "--trust-remote-code",
            "--model-class-name",
            "BertForSequenceClassification",
        ],
    )
    assert result.exit_code == 0
    mock_cli_huggingface_model.assert_called_once_with(
        model_identifier="bert-base-uncased",
        pretrained=True,
        revision="v1.0",
        trust_remote_code=True,
        model_class_name="BertForSequenceClassification",
    )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.commands.utils.models.commands.cli_huggingface_model")
def test_huggingface_command_pretrained_false(mock_cli_huggingface_model, runner):
    # Test with pretrained flag explicitly set to False
    result = runner.invoke(
        models,
        [
            "huggingface",
            "-i",
            "bert-base-uncased",
            "--no-pretrained",  # Explicitly set pretrained to False
        ],
    )
    assert result.exit_code == 0
    mock_cli_huggingface_model.assert_called_once_with(
        model_identifier="bert-base-uncased",
        pretrained=False,
        revision=None,
        trust_remote_code=False,
        model_class_name=None,
    )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.commands.utils.models.commands.cli_huggingface_model")
def test_huggingface_command_pretrained_true(mock_cli_huggingface_model, runner):
    # Test with pretrained flag explicitly set to True
    result = runner.invoke(
        models,
        [
            "huggingface",
            "-i",
            "bert-base-uncased",
            "--pretrained",  # Explicitly set pretrained to True
        ],
    )
    assert result.exit_code == 0
    mock_cli_huggingface_model.assert_called_once_with(
        model_identifier="bert-base-uncased",
        pretrained=True,
        revision=None,
        trust_remote_code=False,
        model_class_name=None,
    )
