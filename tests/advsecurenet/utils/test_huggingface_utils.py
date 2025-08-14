import pytest

from advsecurenet.utils.huggingface_utils import huggingface_general_utils
from advsecurenet.utils.huggingface_utils.huggingface_dataset_utils import (
    check_hub_for_dataset_id,
    verify_hf_dataset_identifier_exists,
)
from advsecurenet.utils.huggingface_utils.huggingface_model_utils import (
    check_hub_for_model_id,
    verify_hf_model_identifier_exists,
)

# --- huggingface_general_utils.py ---


@pytest.mark.parametrize(
    "is_hf_url,hf_url_expected_existence",
    [
        ("https://huggingface.co/microsoft/resnet-18", True),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", True),
        ("https://hf.co/uoft-cs/cifar10", True),
        ("https://example.com/not-hf", False),
        ("", False),
        (None, False),
    ],
)
def test_is_huggingface_url(is_hf_url, hf_url_expected_existence):
    assert (
        huggingface_general_utils.is_huggingface_url(is_hf_url)
        == hf_url_expected_existence
    )


@pytest.mark.parametrize(
    "hf_id,hf_id_expected_existence",
    [
        ("microsoft/resnet-18", True),
        ("uoft-cs/cifar10", True),
        ("invalid-id", False),
        ("", False),
        (None, False),
    ],
)
def test_is_huggingface_id(hf_id, hf_id_expected_existence):
    assert (
        huggingface_general_utils.is_huggingface_id(hf_id) == hf_id_expected_existence
    )


@pytest.mark.parametrize(
    "to_extract_hf_url,extracted_hf_url",
    [
        ("https://huggingface.co/microsoft/resnet-18", "microsoft/resnet-18"),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("https://hf.co/uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("https://example.com/not-hf", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_id_from_url(to_extract_hf_url, extracted_hf_url):
    assert (
        huggingface_general_utils.extract_id_from_url(to_extract_hf_url)
        == extracted_hf_url
    )


@pytest.mark.parametrize(
    "to_process_hf_identifier,processed_hf_identifier",
    [
        ("https://huggingface.co/microsoft/resnet-18", "microsoft/resnet-18"),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("microsoft/resnet-18", "microsoft/resnet-18"),
        ("invalid-id", None),
        ("", None),
        (None, None),
    ],
)
def test_process_hf_identifier(to_process_hf_identifier, processed_hf_identifier):
    assert (
        huggingface_general_utils.process_hf_identifier(to_process_hf_identifier)
        == processed_hf_identifier
    )


# --- huggingface_dataset_hub_utils.py ---


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_exist_hf_dataset_id,dataset_id_expected_existence",
    [
        ("uoft-cs/cifar10", True),
        ("nonexistent-user/nonexistent-dataset", False),
    ],
)
def test_check_hub_for_dataset_id(
    to_exist_hf_dataset_id, dataset_id_expected_existence
):
    assert (
        check_hub_for_dataset_id(to_exist_hf_dataset_id)
        == dataset_id_expected_existence
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_exist_hf_dataset_identifier,dataset_identifier_expected_existence",
    [
        ("uoft-cs/cifar10", True),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", True),
        ("https://hf.co/datasets/uoft-cs/cifar10", True),
        ("nonexistent-user/nonexistent-dataset", False),
        ("https://huggingface.co/datasets/nonexistent-user/nonexistent-dataset", False),
        ("https://hf.co/datasets/nonexistent-user/nonexistent-dataset", False),
        ("invalid-id", False),
        ("", False),
        (None, False),
    ],
)
def test_verify_hf_dataset_identifier_exists(
    to_exist_hf_dataset_identifier, dataset_identifier_expected_existence
):
    assert (
        verify_hf_dataset_identifier_exists(to_exist_hf_dataset_identifier)
        == dataset_identifier_expected_existence
    )


# --- huggingface_model_hub_utils.py ---


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_check_model_id,model_id_expected_existence",
    [
        ("microsoft/resnet-18", True),
        ("bert-base-uncased", True),
        ("nonexistent-user/nonexistent-model", False),
    ],
)
def test_check_hub_for_model_id(to_check_model_id, model_id_expected_existence):
    assert check_hub_for_model_id(to_check_model_id) == model_id_expected_existence


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_verify_hf_model_identifier,hf_model_identifier_expected_existence",
    [
        ("microsoft/resnet-18", True),
        ("https://huggingface.co/microsoft/resnet-18", True),
        ("google-bert/bert-base-uncased", True),
        ("https://hf.co/google-bert/bert-base-uncased", True),
        ("nonexistent-user/nonexistent-model", False),
        ("https://huggingface.co/nonexistent-user/nonexistent-model", False),
        ("invalid-id", False),
        ("", False),
        (None, False),
    ],
)
def test_verify_hf_model_identifier_exists(
    to_verify_hf_model_identifier, hf_model_identifier_expected_existence
):
    assert (
        verify_hf_model_identifier_exists(to_verify_hf_model_identifier)
        == hf_model_identifier_expected_existence
    )
