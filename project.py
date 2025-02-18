# from ultralytics import YOLO
# import cv2
# import torch
# import numpy as np
# from transformers import ViTFeatureExtractor, ViTModel
# from PIL import Image

# # Load pre-trained YOLOv8 segmentation model
# yolo_model = YOLO("yolov8n-seg.pt")  # Segmentation model

# # Load pre-trained ViT model for feature extraction
# vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# # Load an image
# image_path = "C:/Users/PMLS/Desktop/xloop_project/deepfashion2/test/test/image/000001.jpg"  # Change this to your image path
# image = cv2.imread(image_path)

# # Perform inference with YOLOv8
# results = yolo_model(image)

# cropped_images = []  # To store cropped garment images
# features_list = []  # To store extracted features

# # Process YOLO segmentation results
# for result in results:
#     masks = result.masks  # Segmentation masks
#     boxes = result.boxes  # Bounding boxes
    
#     for mask, box in zip(masks.xy, boxes.xyxy):
#         x1, y1, x2, y2 = map(int, box)

#         # Crop the detected garment
#         cropped_img = image[y1:y2, x1:x2]
#         cropped_images.append(cropped_img)

#         # Convert cropped image to PIL format for ViT
#         cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

#         # Extract ViT features
#         inputs = vit_extractor(images=cropped_pil, return_tensors="pt")
#         with torch.no_grad():
#             outputs = vit_model(**inputs)
#             features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract CLS token

#         features_list.append(features)

#         # Draw bounding box
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # Show image with segmentation
# cv2.imshow("Segmented Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Print extracted features (for debugging)
# print("Extracted Features Shape:", np.array(features_list).shape)  # (num_garments, feature_dim)

# from ultralytics import YOLO
# import cv2
# import torch
# import numpy as np
# import faiss
# from transformers import ViTFeatureExtractor, ViTModel
# from PIL import Image
# from collections import Counter
# import matplotlib.pyplot as plt

# # Load pre-trained models
# yolo_model = YOLO("yolov8n-seg.pt")  # YOLOv8 segmentation
# vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# # FAISS index setup
# feature_dim = 768  # ViT feature vector size
# index = faiss.IndexFlatL2(feature_dim)

# # Load an image
# image_path = "C:/Users/PMLS/Desktop/xloop_project/deepfashion2/test/test/image/000001.jpg"
# image = cv2.imread(image_path)

# # Perform segmentation with YOLOv8
# results = yolo_model(image)
# cropped_images = []
# features_list = []

# # Process YOLO results
# for result in results:
#     masks = result.masks  # Segmentation masks
#     boxes = result.boxes  # Bounding boxes

#     for mask, box in zip(masks.xy, boxes.xyxy):
#         x1, y1, x2, y2 = map(int, box)

#         # Crop detected garment
#         cropped_img = image[y1:y2, x1:x2]
#         cropped_images.append(cropped_img)

#         # Convert to PIL format for ViT feature extraction
#         cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
#         inputs = vit_extractor(images=cropped_pil, return_tensors="pt")

#         # Extract ViT features
#         with torch.no_grad():
#             outputs = vit_model(**inputs)
#             features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

#         features_list.append(features)
#         index.add(np.expand_dims(features, axis=0))  # Add to FAISS index

#         # Draw bounding box
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # Show image with segmentation
# cv2.imshow("Segmented Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Convert features list to numpy array for FAISS
# features_db = np.array(features_list).astype('float32')

# # Store features in FAISS index
# index.add(features_db)

# ### Function for color matching ###
# def extract_dominant_colors(image, k=3):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     pixels = image.reshape(-1, 3)
    
#     # K-Means Clustering for dominant colors
#     kmeans = faiss.Kmeans(3, k)
#     kmeans.train(pixels.astype(np.float32))
#     _, labels = kmeans.index.search(pixels.astype(np.float32), 1)

#     # Count occurrences of each cluster
#     color_counts = Counter(labels.flatten())
#     dominant_colors = [kmeans.centroids[i] for i in color_counts.keys()]
    
#     return dominant_colors

# # Example: Extract colors from cropped images
# for idx, cropped in enumerate(cropped_images):
#     colors = extract_dominant_colors(cropped, k=3)
#     print(f"Garment {idx+1} Dominant Colors: {colors}")

# # FAISS Similarity Search
# query_feature = features_list[0].reshape(1, -1)  # Example query (first item)
# D, I = index.search(query_feature, 5)  # Retrieve 5 nearest neighbors

# print("Top 5 Similar Items:", I)




from ultralytics import YOLO
import cv2
import torch
import numpy as np
import faiss
import pickle  # For database storage
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from collections import Counter
import os

# Load pre-trained models
yolo_model = YOLO("yolov8n-seg.pt")  # YOLOv8 segmentation
vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# FAISS index setup
feature_dim = 768  # ViT feature vector size
index = faiss.IndexFlatL2(feature_dim)

# Database storage (features & colors)
database = {"features": [], "colors": [], "image_paths": []}


### **Function to extract dominant colors** ###
def extract_dominant_colors(image, k=3):
    """Extracts k dominant colors from an image using k-means clustering."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    # K-Means Clustering for dominant colors
    kmeans = faiss.Kmeans(3, k)
    kmeans.train(pixels.astype(np.float32))
    _, labels = kmeans.index.search(pixels.astype(np.float32), 1)

    # Count occurrences of each cluster
    color_counts = Counter(labels.flatten())
    dominant_colors = [kmeans.centroids[i] for i in color_counts.keys()]
    
    return dominant_colors


### **Function to process a dataset folder** ###
def process_dataset(dataset_folder):
    """Processes all images in the dataset, extracts features & colors, and stores them."""
    for file_name in os.listdir(dataset_folder):
        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        image_path = os.path.join(dataset_folder, file_name)
        image = cv2.imread(image_path)
        
        # Perform segmentation with YOLOv8
        results = yolo_model(image)
        
        for result in results:
            boxes = result.boxes  # Bounding boxes
            
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cropped_img = image[y1:y2, x1:x2]  # Crop garment

                # Convert to PIL format for ViT
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                inputs = vit_extractor(images=cropped_pil, return_tensors="pt")

                # Extract ViT features
                with torch.no_grad():
                    outputs = vit_model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

                # Store features & colors
                dominant_colors = extract_dominant_colors(cropped_img, k=3)
                index.add(np.expand_dims(features, axis=0))  # Add feature to FAISS

                # Add to database
                database["features"].append(features)
                database["colors"].append(dominant_colors)
                database["image_paths"].append(image_path)

    # Save database
    with open("fashion_database.pkl", "wb") as f:
        pickle.dump(database, f)


### **Function to find similar products** ###
def find_similar(image_path, top_k=5):
    """Finds top_k most similar garments to the given query image."""
    image = cv2.imread(image_path)
    results = yolo_model(image)
    
    for result in results:
        boxes = result.boxes  # Bounding boxes

        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = image[y1:y2, x1:x2]  # Crop detected garment

            # Convert to PIL format for ViT
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            inputs = vit_extractor(images=cropped_pil, return_tensors="pt")

            # Extract ViT features
            with torch.no_grad():
                outputs = vit_model(**inputs)
                query_feature = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

            # Search FAISS index
            query_feature = np.expand_dims(query_feature, axis=0)
            D, I = index.search(query_feature, top_k)

            print("Top", top_k, "Similar Products:")
            for idx in I[0]:
                print("Image Path:", database["image_paths"][idx])
                print("Colors:", database["colors"][idx])

    return I[0]  # Return indices of similar products


# **PROCESS DATASET (Run once to store database)**
dataset_path = "C:/Users/PMLS/Desktop/xloop_project/deepfashion2/test/test/image"
process_dataset(dataset_path)

# **FIND SIMILAR PRODUCTS (Example query)**
query_image_path = "C:/Users/PMLS/Desktop/xloop_project/deepfashion2/test/test/image/000001.jpg"
similar_indices = find_similar(query_image_path, top_k=5)
