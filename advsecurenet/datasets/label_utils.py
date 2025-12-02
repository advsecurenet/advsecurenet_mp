from typing import Any, Dict, List, Optional

from advsecurenet.datasets.COCO.coco_utils import (
    COCO_INSTANCE_CATEGORY_NAMES,
    ID_TO_CONTIGUOUS,
)
from advsecurenet.datasets.PascalVOC.pascalvoc_utils import PASCAL_VOC_CATEGORY_NAMES


_LABELS_REGISTRY: Dict[str, List[str]] = {
    "COCO": COCO_INSTANCE_CATEGORY_NAMES,
    "PASCAL_VOC": PASCAL_VOC_CATEGORY_NAMES,
}


def get_dataset_classes_count(name: str) -> int:
    """
    Return number of classes for a dataset; 0 if unknown.
    """
    return len(_LABELS_REGISTRY.get((name or "").upper(), []))


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


# ---- COCO -> Pascal VOC mapping ----
def _normalize_name(name: str) -> str:
    s = (name or "").strip().lower()
    for ch in (" ", "-", "_", "/", "\\"):
        s = s.replace(ch, "")
    return s


def _build_pascal_index() -> Dict[str, int]:
    return {_normalize_name(n): i for i, n in enumerate(PASCAL_VOC_CATEGORY_NAMES)}


def _build_coco_to_pascal_mapping() -> Dict[int, int]:
    pascal_idx = _build_pascal_index()
    synonyms = {
        "airplane": "aeroplane",
        "motorcycle": "motorbike",
        "couch": "sofa",
        "pottedplant": "pottedplant",
        "diningtable": "diningtable",
        "tv": "tvmonitor",
    }
    mapping: Dict[int, int] = {}
    for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES):
        norm = _normalize_name(name)
        norm = synonyms.get(norm, norm)
        if norm in pascal_idx:
            mapping[i] = pascal_idx[norm]
    return mapping


_COCO_TO_PASCAL_MAPPING: Dict[int, int] = _build_coco_to_pascal_mapping()


def coco_label_id_to_pascal(
    label_id: int, *, assume_contiguous: bool = True, unmapped_value: int = -1
) -> int:
    if label_id is None:
        return unmapped_value
    if not assume_contiguous:
        if label_id not in ID_TO_CONTIGUOUS:
            return unmapped_value
        label_id = ID_TO_CONTIGUOUS[label_id]
    return _COCO_TO_PASCAL_MAPPING.get(int(label_id), unmapped_value)


def coco_label_ids_to_pascal(
    label_ids: List[int],
    *,
    assume_contiguous: bool = True,
    unmapped_value: int = -1,
    drop_unmapped: bool = False
) -> List[int]:
    out: List[int] = []
    for lid in label_ids:
        mapped = coco_label_id_to_pascal(
            lid, assume_contiguous=assume_contiguous, unmapped_value=unmapped_value
        )
        if drop_unmapped and mapped == unmapped_value:
            continue
        out.append(mapped)
    return out
