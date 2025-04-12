import torch
import ultralytics

# Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

# Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
img = "https://ultralytics.com/images/zidane.jpg"  # Example image

# Perform inference (handles batching, resizing, normalization automatically)
results = model(img)

# Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
results.print()  # Print results to console
results.show()  # Display results in a window

targets = []
for i in range(len(results.xyxy)):  # Iterate over each image in the batch
    boxes = results.xyxy[i][:, :4].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    labels = results.xyxy[i][:, 5].cpu().numpy().astype(int)  # Class labels
    scores = results.xyxy[i][:, 4].cpu().numpy()  # Confidence scores

    # Construct the target dictionary for the current image
    target = {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
    }
    targets.append(target)

target_boxes = [target["boxes"] for target in targets]  # List of bounding boxes for each image
target_labels = [target["labels"] for target in targets]  # List of labels for each image
pred_boxes = [y["boxes"] for y in y_list]
pred_labels = [y["labels"] for y in y_list]

# Classification loss
classification_loss = self.classification_loss_fn(pred_labels, target_labels)
# Bounding box regression loss
bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes)

# Combine losses
total_loss = classification_loss + bbox_loss

print(total_loss)