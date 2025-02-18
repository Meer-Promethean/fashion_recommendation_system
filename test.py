from ultralytics import YOLO

# Load your model (YOLOv8 model)
model = YOLO('yolov8m-seg.pt')  # or your custom model path

# Start training with verbose mode enabled
model.train(
    data='./deepfashion2/dataset.yaml',  # Path to dataset.yaml
    epochs=100,
    batch=32,
    imgsz=640,
    verbose=True
)

