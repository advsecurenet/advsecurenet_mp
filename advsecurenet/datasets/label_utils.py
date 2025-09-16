from typing import Any, Dict, List, Optional

from advsecurenet.datasets.COCO.coco_utils import COCO_INSTANCE_CATEGORY_NAMES
from advsecurenet.datasets.PascalVOC.pascalvoc_utils import PASCAL_VOC_CATEGORY_NAMES


_LABELS_REGISTRY: Dict[str, List[str]] = {
    "COCO": COCO_INSTANCE_CATEGORY_NAMES,
    "PASCAL_VOC": PASCAL_VOC_CATEGORY_NAMES,
}


def get_dataset_labels(name: str) -> List[str]:
    """
    Return label list for a dataset; empty list if unknown.
    """
    return _LABELS_REGISTRY.get((name or "").upper(), [])


def get_model_label_names(model: Any) -> List[str]:
    """
    Try to extract label names from a model instance:
      - HuggingFace: model.config.id2label
      - YOLO-style: model.names
    Returns [] if none found.
    """
    cfg = getattr(model, "config", None) or getattr(
        getattr(model, "model", None), "config", None
    )
    id2label = getattr(cfg, "id2label", None)
    if isinstance(id2label, dict) and id2label:
        max_id = max(int(k) for k in id2label.keys())
        return [str(id2label.get(i, str(i))) for i in range(max_id + 1)]
    names = getattr(model, "names", None) or getattr(
        getattr(model, "model", None), "names", None
    )
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return []


def resolve_label_names(
    dataset_name: Optional[str] = None, model: Optional[Any] = None
) -> List[str]:
    """
    Prefer model-provided label names, else fall back to dataset defaults, else [].
    """
    names = []
    if model is not None:
        names = get_model_label_names(model)
    if not names and dataset_name:
        names = get_dataset_labels(dataset_name)
    return names or []
