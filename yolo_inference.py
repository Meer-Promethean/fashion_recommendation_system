# yolo_inference.py
import os
import cv2
from ultralytics import YOLO

# Set the path to your trained YOLO model file.
yolo_model_path = './output2/yolov8_trained.pt'
yolo_model = YOLO(yolo_model_path)

def detect_and_crop(image_path, output_dir="detected_products", conf_threshold=0.2):
    """
    Runs YOLO inference on an image, extracts bounding boxes for detected objects,
    and saves each crop as a separate image.
    
    Args:
      image_path (str): Path to the input image.
      output_dir (str): Directory where cropped images are saved.
      conf_threshold (float): Confidence threshold for detections.
      
    Returns:
      List[str]: Paths to the saved cropped images.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image from path: {image_path}")
    
    results = yolo_model.predict(source=image_path, conf=conf_threshold)
    cropped_paths = []
    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Expected format: [x1, y1, x2, y2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            crop_path = os.path.join(output_dir, f"crop_{i}.jpg")
            cv2.imwrite(crop_path, crop)
            cropped_paths.append(crop_path)
    else:
        print("No detections found.")
    return cropped_paths
