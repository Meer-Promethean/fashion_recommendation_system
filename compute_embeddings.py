# compute_embeddings.py
import os
import numpy as np
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
from database import SessionLocal, Product

# Load pre-trained ViT components
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

def extract_vit_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = vit_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    # Use the [CLS] token embedding as the global feature vector
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

def compute_catalog_embeddings(catalog_dir, embedding_file, max_images=500):
    session = SessionLocal()
    products = session.query(Product).all()
    # If the database is empty, add images from catalog_dir
    if not products:
        image_files = [f for f in os.listdir(catalog_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))][:max_images]
        for file in image_files:
            image_path = os.path.join(catalog_dir, file)
            prod = Product(name=file, image_path=image_path, embedding_index=None)
            session.add(prod)
        session.commit()
        products = session.query(Product).all()
    
    embeddings_list = []
    for idx, prod in enumerate(tqdm(products, desc="Computing catalog embeddings")):
        try:
            emb = extract_vit_features(prod.image_path)
            embeddings_list.append(emb)
            prod.embedding_index = idx
        except Exception as e:
            print(f"Error processing {prod.image_path}: {e}")
    session.commit()
    session.close()
    
    catalog_embeddings = np.array(embeddings_list, dtype='float32')
    np.save(embedding_file, catalog_embeddings)
    print("Catalog embeddings computed and saved to", embedding_file)

if __name__ == "__main__":
    catalog_dir = "./dbimages"  # Folder with your product images
    embedding_file = "./deepfashion2catalog_embeddings.npy"
    compute_catalog_embeddings(catalog_dir, embedding_file)
