import string
import torch
import numpy as np
import datasets
from datasets import load_dataset
from advsecurenet.computer_vision.object_detection.attacks.adversarial_patch_based.dpatch import DPatch

COCO_LABEL_MAP = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic",
    11: "fire",
    12: "street",
    13: "stop",
    14: "parking",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports",
    38: "kite",
    39: "baseball",
    40: "baseball",
    41: "skateboard",
    42: "surfboard",
    43: "tennis",
    44: "bottle",
    45: "plate",
    46: "wine",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted",
    65: "bed",
    66: "mirror",
    67: "dining",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy",
    89: "hair",
    90: "toothbrush",
    91: "hair",
    92: "banner",
    93: "blanket",
    94: "branch",
    95: "bridge",
    96: "building",
    97: "bush",
    98: "cabinet",
    99: "cage",
    100: "cardboard",
    101: "carpet",
    102: "ceiling",
    103: "ceiling",
    104: "cloth",
    105: "clothes",
    106: "clouds",
    107: "counter",
    108: "cupboard",
    109: "curtain",
    110: "desk",
    111: "dirt",
    112: "door",
    113: "fence",
    114: "floor",
    115: "floor",
    116: "floor",
    117: "floor",
    118: "floor",
    119: "flower",
    120: "fog",
    121: "food",
    122: "fruit",
    123: "furniture",
    124: "grass",
    125: "gravel",
    126: "ground",
    127: "hill",
    128: "house",
    129: "leaves",
    130: "light",
    131: "mat",
    132: "metal",
    133: "mirror",
    134: "moss",
    135: "mountain",
    136: "mud",
    137: "napkin",
    138: "net",
    139: "paper",
    140: "pavement",
    141: "pillow",
    142: "plant",
    143: "plastic",
    144: "platform",
    145: "playingfield",
    146: "railing",
    147: "railroad",
    148: "river",
    149: "road",
    150: "rock",
    151: "roof",
    152: "rug",
    153: "salad",
    154: "sand",
    155: "sea",
    156: "shelf",
    157: "sky",
    158: "skyscraper",
    159: "snow",
    160: "solid",
    161: "stairs",
    162: "stone",
    163: "straw",
    164: "structural",
    165: "table",
    166: "tent",
    167: "textile",
    168: "towel",
    169: "tree",
    170: "vegetable",
    171: "wall",
    172: "wall",
    173: "wall",
    174: "wall",
    175: "wall",
    176: "wall",
    177: "wall",
    178: "water",
    179: "waterdrops",
    180: "window",
    181: "window",
    182: "wood"
}

def preprocess_coco_sample(sample):
        """
        Preprocess a single COCO dataset sample to make it compatible with the DPatch attack code.

        Args:
            sample (dict): A single sample from the COCO dataset.

        Returns:
            dict: Processed sample with image array and annotations.
        """
        # Load image and convert to numpy array
        image = sample["image"]  # PIL.Image
        image_array = np.asarray(image, dtype=np.float32) #/ 255.0  # Normalize to [0, 1]

        # Extract bounding boxes and labels
        bboxes = sample["objects"]["bbox"]  # List of bounding boxes
        labels = sample["objects"]["category"]  # List of category labels

        # Convert bounding boxes to numpy array (x1, y1, x2, y2)
        bboxes_array = np.array(
            [[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bboxes], dtype=np.float32
        )

        # Convert labels to numpy array
        labels_array = [COCO_LABEL_MAP[label] for label in labels]

        return {
            "image": image_array,
            "annotations": {
                "boxes": bboxes_array,
                "labels": labels_array,
            },
        }

def load_dataset_my(dataset: string) -> None:
        #dataset = load_dataset("detection-datasets/coco", split="val")
        dataset = load_dataset("detection-datasets/coco")
        # Preprocess the dataset to make it compatible with the DPatch attack code
        coco_dataset = [preprocess_coco_sample(sample) for sample in dataset["val"]]   
        return coco_dataset

coco_dataset = load_dataset_my("coco")
#print(coco_dataset[0])
object_detector = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
res = object_detector(coco_dataset[0]["image"])
res.show()
print("detected")
print(res)
print("true")
print(coco_dataset[0]["annotations"]["labels"])
# predictions = []
# for i in range(len(res.xyxy)):  # Iterate over each image in the batch
#                     boxes = res.xyxy[i][:, :4].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
#                     labels = res.xyxy[i][:, 5].cpu().numpy().astype(int)  # Class labels
#                     scores = res.xyxy[i][:, 4].cpu().numpy()  # Confidence scores
                    
#                     target = {
#                         "boxes": boxes,
#                         "labels": labels,
#                         "scores": scores,
#                     }
#                     predictions.append(target)

# print("CCC")
# print(predictions[0])
# predictions = [COCO_LABEL_MAP[label] for label in predictions[0]["labels"]]
# print(predictions)

# print("AAA")
# print(predictions)
