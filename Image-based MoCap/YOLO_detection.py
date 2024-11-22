from ultralytics import YOLO
import cv2
import json
import torch

### DETECTION

# Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

# # Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")
# # model = YOLO("/teamspace/studios/this_studio/runs/detect/train5/weights/best.pt") ### FOR VALIDATION

# # Train the model using the 'coco8.yaml' dataset for 3 epochs
# results = model.train(data="coco8.yaml", epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()   ### ONLY for Validation Metrics

# # Perform object detection on an image using the model
results = model("/teamspace/studios/this_studio/000000006460.jpg",save=True)

quit()
image = cv2.imread("/teamspace/studios/this_studio/COCO_sport_samples/000000007278.jpg")
H, W, _ = image.shape

bounding_boxes = []

for result in results:
    for box in result.boxes:
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = box.xyxy[0]  # Coordinates in the form [x_min, y_min, x_max, y_max]
        
        # Get the class label and confidence score
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Append bounding box data to the list
        bounding_boxes.append({
            'class_id': class_id,
            'confidence': confidence,
            'bounding_box': {
                'x_min': float(x_min),
                'y_min': float(y_min),
                'x_max': float(x_max),
                'y_max': float(y_max)
            }
        })

# Save the bounding box data to a JSON file
json_file_path = 'bounding_boxes.json'
with open(json_file_path, 'w') as f:
    json.dump(bounding_boxes, f, indent=4)

print(f"Bounding box data saved to {json_file_path}")

### SAVE THE RESULTS only for one class
bounding_boxes_selected = []

for result in results:
        boxes = result.boxes.data
        clss = boxes[:, 5]
        selected_indices = torch.where(clss == 37)

        boxes = result.boxes[selected_indices]

        for box in boxes:
            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Coordinates in the form [x_min, y_min, x_max, y_max]
            
            # Get the class label and confidence score
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Append bounding box data to the list
            bounding_boxes_selected.append({
                'class_id': class_id,
                'confidence': confidence,
                'bounding_box': {
                    'x_min': float(x_min),
                    'y_min': float(y_min),
                    'x_max': float(x_max),
                    'y_max': float(y_max)
                }
            })

# Save the bounding box data to a JSON file
json_file_path = 'bounding_boxes_selected.json'
with open(json_file_path, 'w') as f:
    json.dump(bounding_boxes_selected, f, indent=4)

print(f"Bounding box data saved to {json_file_path}")


# Export the model to ONNX format
success = model.export(format="onnx") ### TO SAVE THE MODEL ONLY FOR TRAINING


### Tracking

# Configure the tracking parameters and run the tracker
model = YOLO("yolo11n.pt")
results = model.track(source="/teamspace/studios/this_studio/input_video.mp4",save=True)



