from typing import Dict, List

# Pascal VOC (20 classes)
PASCAL_VOC_CATEGORY_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]

# raw ids 1..20 (like many VOC codebases); background would be 0 if you ever need it
PASCAL_VOC_CATEGORY_IDS = list(range(1, 21))
NAME_TO_RAW_ID = {n: i + 1 for i, n in enumerate(PASCAL_VOC_CATEGORY_NAMES)}
RAW_ID_TO_NAME = {v: k for k, v in NAME_TO_RAW_ID.items()}


def _normalize_name(name: str) -> str:
    name = (name or "").strip().lower()
    if name in {"tv/monitor", "tv / monitor"}:
        return "tvmonitor"
    return name


def voc_to_coco_anns(target: Dict) -> List[Dict]:
    """
    Convert torchvision VOCDetection 'target' (XML-as-dict) into a list of
    COCO-like annotation dicts (to keep target type consistent).
    """
    ann = target.get("annotation", {})
    size = ann.get("size", {})
    W = float(size.get("width", 0) or 0)
    H = float(size.get("height", 0) or 0)
    objs = ann.get("object", [])
    if not isinstance(objs, list):
        objs = [objs] if objs else []
    out: List[Dict] = []
    for obj in objs:
        name = _normalize_name(obj.get("name"))
        if name not in NAME_TO_RAW_ID:
            continue
        bb = obj.get("bndbox", {})
        try:
            xmin = float(bb["xmin"]); ymin = float(bb["ymin"])
            xmax = float(bb["xmax"]); ymax = float(bb["ymax"])
        except Exception:
            continue
        # clip to image bounds if size is known
        if W > 0 and H > 0:
            xmin = max(0.0, min(xmin, W))
            xmax = max(0.0, min(xmax, W))
            ymin = max(0.0, min(ymin, H))
            ymax = max(0.0, min(ymax, H))
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        difficult = obj.get("difficult", "0")
        try:
            difficult = int(difficult)
        except Exception:
            difficult = 0
        out.append(
            {
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "category_id": int(NAME_TO_RAW_ID[name] - 1),
                "area": float(w * h),
                "iscrowd": 0,
                "difficult": difficult,
            }
        )
    return out
