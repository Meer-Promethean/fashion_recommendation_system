# app.py

import os
import uvicorn
import nest_asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Import your project modules.
from database import SessionLocal, Product
from yolo_inference import detect_and_crop
from compute_embeddings import extract_vit_features
from faiss_search import build_faiss_index, search_similar_products

# Build FAISS index from your precomputed catalog embeddings file.
# (This file is generated by running compute_embeddings.py.)
embedding_file = "./deepfashion2catalog_embeddings.npy"
index = build_faiss_index(embedding_file)

# Create the FastAPI app instance.
app = FastAPI()

@app.post("/detect_and_recommend/")
async def detect_and_recommend(file: UploadFile = File(...)):
    # Save the uploaded file using an absolute path.
    temp_input = os.path.join(os.getcwd(), "temp_input.jpg")
    data = await file.read()
    print("Uploaded data length:", len(data))
    if len(data) == 0:
        return JSONResponse(content={"error": "Uploaded file is empty."}, status_code=400)
    with open(temp_input, "wb") as f:
        f.write(data)
    
    # Debug: print the file size.
    file_size = os.path.getsize(temp_input)
    print("File saved at:", temp_input, "Size:", file_size)
    
    # Verify that the saved file is a valid image using PIL.
    from PIL import Image
    try:
        img = Image.open(temp_input)
        img.verify()  # Raises an exception if the image is invalid.
        print("temp_input.jpg is a valid image.")
    except Exception as e:
        return JSONResponse(content={"error": "Uploaded file is not a valid image: " + str(e)}, status_code=400)
    
    # Run detection and cropping using your YOLO-based function.
    crops = detect_and_crop(temp_input, output_dir="detected_products")
    
    recommendations = {}
    session = SessionLocal()
    
    # For each crop, extract features, perform similarity search, and look up products in the DB.
    for crop_path in crops:
        try:
            embedding = extract_vit_features(crop_path)
            distances, indices = search_similar_products(index, embedding, k=5)
            similar_products = []
            for idx in indices[0]:
                prod = session.query(Product).filter(Product.embedding_index == int(idx)).first()
                if prod:
                    similar_products.append({
                        "id": prod.id,
                        "name": prod.name,
                        "image_path": prod.image_path,
                        "embedding_index": prod.embedding_index
                    })
            recommendations[crop_path] = {
                "similar_products": similar_products,
                "distances": distances.tolist()
            }
        except Exception as e:
            recommendations[crop_path] = {"error": str(e)}
    
    session.close()
    return JSONResponse(content={"detected_crops": crops, "recommendations": recommendations})

if __name__ == "__main__":
    # Patch the asyncio loop if needed (e.g., for Colab; on local systems this may be optional).
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
