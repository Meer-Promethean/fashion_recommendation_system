# Fashion Recommender System

This repository contains a fashion recommender system that leverages YOLO for object detection, a Vision Transformer (ViT) for feature extraction, and FAISS for fast similarity search.
The system provides a catalog view and an upload interface for real-time recommendations.

## Project Overview

- **Frontend:** Built with Streamlit to display product catalogs and enable image uploads.
- **Backend:** Powered by FastAPI to process images, detect clothing items using YOLO, extract features with ViT, and perform similarity search using FAISS.
- **Database:** SQLite is used to store product details.
- **Embeddings:** Precomputed embeddings are generated using ViT and stored for fast retrieval.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/fashion-recommender.git
   cd fashion-recommender

trained model output: put it in folder named output2:
https://drive.google.com/file/d/1n-waVbSVX2vkV-Q0HSsHE7qtc_cxuUu9/view?usp=drive_link
