from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
results = model("/teamspace/studios/this_studio/COCO_sport_samples/000000037988.jpg",save=True) 