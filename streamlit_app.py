# streamlit_app.py

import streamlit as st
import sqlite3
import requests
from PIL import Image
import io

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Fashion Recommender", layout="wide")

# -------------------------------
# Function to fetch products from DB
# -------------------------------
def fetch_products():
    # Connect to SQLite DB (assuming products.db is in the project folder)
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, image_path FROM products")
    products = cursor.fetchall()
    conn.close()
    return products

# -------------------------------
# Create Sidebar Menu for Navigation
# -------------------------------
menu = st.sidebar.selectbox("Menu", ["Catalog", "Upload & Recommendations"])

# -------------------------------
# Catalog Tab: View Product Catalog
# -------------------------------
if menu == "Catalog":
    st.header("Product Catalog")
    products = fetch_products()
    
    if products:
        # We'll display images in a grid.
        cols = st.columns(4)  # Change number of columns as desired.
        for i, (prod_id, name, image_path) in enumerate(products):
            with cols[i % 4]:
                try:
                    # Load and display image
                    image = Image.open(image_path)
                    st.image(image, caption=name, use_container_width=True)
                    st.text(f"ID: {prod_id}")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    else:
        st.warning("No products found in the database.")

# -------------------------------
# Upload & Recommendations Tab
# -------------------------------
if menu == "Upload & Recommendations":
    st.header("Upload an Image for Recommendations")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Prepare file for POST request.
        files = {"file": ("uploaded_image.jpg", uploaded_file.getvalue(), "image/jpeg")}
        
        with st.spinner("Processing..."):
            response = requests.post(f"{API_URL}/detect_and_recommend/", files=files)
        
        if response.status_code == 200:
            data = response.json()
            st.success("Processing complete!")
            detected_crops = data.get("detected_crops", [])
            recommendations = data.get("recommendations", {})
            
            if detected_crops:
                st.subheader("Detected Clothing Items")
                for crop in detected_crops:
                    try:
                        st.image(crop, caption=f"Detected: {crop}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not load crop image: {e}")
            else:
                st.info("No clothing items detected in the uploaded image.")
            
            if recommendations:
                st.subheader("Recommendations for Detected Items")
                for crop, rec in recommendations.items():
                    st.write(f"**For detected crop:** {crop}")
                    if "error" in rec:
                        st.error(rec["error"])
                    else:
                        cols = st.columns(4)  # Show recommendations in a grid of 4 columns.
                        for i, prod in enumerate(rec["similar_products"]):
                            with cols[i % 4]:
                                try:
                                    rec_image = Image.open(prod["image_path"])
                                    st.image(rec_image, caption=prod["name"], use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Error loading recommended image: {e}")
        else:
            st.error("Error processing the image on the backend.")


# import streamlit as st
# import sqlite3
# import requests
# from PIL import Image
# import io

# # Backend API URL
# API_URL = "http://127.0.0.1:8000"

# st.set_page_config(page_title="Fashion Recommender", layout="wide")

# # Inject custom CSS for an e-commerce feel
# st.markdown("""
# <style>
# body {
#     font-family: 'Arial', sans-serif;
#     background-color: #f8f8f8;
# }
# .header {
#     text-align: center;
#     padding: 20px;
# }
# .product-card {
#     background-color: #fff;
#     border: 1px solid #e1e1e1;
#     border-radius: 8px;
#     padding: 15px;
#     margin: 10px;
#     box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     text-align: center;
# }
# .product-img {
#     width: 100%;
#     height: auto;
#     border-radius: 5px;
#     margin-bottom: 10px;
# }
# .product-name {
#     font-size: 16px;
#     font-weight: bold;
#     margin-bottom: 5px;
# }
# .product-price {
#     font-size: 14px;
#     color: #e74c3c;
# }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # Function to fetch products from DB
# # -------------------------------
# def fetch_products():
#     # Connect to SQLite DB (assuming products.db is in the project folder)
#     conn = sqlite3.connect("products.db")
#     cursor = conn.cursor()
#     # Updated query for the new schema
#     cursor.execute("SELECT id, name, image_url, price FROM products")
#     products = cursor.fetchall()
#     conn.close()
#     return products

# # Dummy values if product details are missing
# DUMMY_IMAGE = "https://via.placeholder.com/350x350?text=No+Image"
# DUMMY_NAME = "Dummy Product Name"
# DUMMY_PRICE = "Price Not Available"

# # -------------------------------
# # Create Sidebar Menu for Navigation
# # -------------------------------
# menu = st.sidebar.selectbox("Menu", ["Catalog", "Upload & Recommendations"])

# # -------------------------------
# # Catalog Tab: View Product Catalog
# # -------------------------------
# if menu == "Catalog":
#     st.markdown('<div class="header"><h1>Product Catalog</h1></div>', unsafe_allow_html=True)
#     products = fetch_products()
    
#     if products:
#         # Display products in a grid (4 columns)
#         cols = st.columns(4)
#         for i, (prod_id, name, image_url, price) in enumerate(products):
#             # Replace missing fields with dummy values
#             if not image_url:
#                 image_url = DUMMY_IMAGE
#             if not name:
#                 name = DUMMY_NAME
#             if not price:
#                 price = DUMMY_PRICE

#             with cols[i % 4]:
#                 st.markdown(f"""
#                 <div class="product-card">
#                     <img class="product-img" src="{image_url}" alt="{name}">
#                     <div class="product-name">{name}</div>
#                     <div class="product-price">{price}</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#     else:
#         st.warning("No products found in the database.")

# # -------------------------------
# # Upload & Recommendations Tab
# # -------------------------------
# if menu == "Upload & Recommendations":
#     st.markdown('<div class="header"><h1>Upload an Image for Recommendations</h1></div>', unsafe_allow_html=True)
    
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_container_width=True)
        
#         # Prepare file for POST request.
#         files = {"file": ("uploaded_image.jpg", uploaded_file.getvalue(), "image/jpeg")}
        
#         with st.spinner("Processing..."):
#             response = requests.post(f"{API_URL}/detect_and_recommend/", files=files)
        
#         if response.status_code == 200:
#             data = response.json()
#             st.success("Processing complete!")
#             detected_crops = data.get("detected_crops", [])
#             recommendations = data.get("recommendations", {})
            
#             if detected_crops:
#                 st.subheader("Detected Clothing Items")
#                 for crop in detected_crops:
#                     try:
#                         st.image(crop, caption=f"Detected: {crop}", use_container_width=True)
#                     except Exception as e:
#                         st.warning(f"Could not load crop image: {e}")
#             else:
#                 st.info("No clothing items detected in the uploaded image.")
            
#             if recommendations:
#                 st.subheader("Recommendations for Detected Items")
#                 for crop, rec in recommendations.items():
#                     st.write(f"**For detected crop:** {crop}")
#                     if "error" in rec:
#                         st.error(rec["error"])
#                     else:
#                         rec_cols = st.columns(4)  # Grid for recommended products
#                         for i, prod in enumerate(rec["similar_products"]):
#                             with rec_cols[i % 4]:
#                                 try:
#                                     rec_image = Image.open(prod["image_path"])
#                                     st.image(rec_image, caption=prod["name"], use_container_width=True)
#                                 except Exception as e:
#                                     st.warning(f"Error loading recommended image: {e}")
#         else:
#             st.error("Error processing the image on the backend.")
