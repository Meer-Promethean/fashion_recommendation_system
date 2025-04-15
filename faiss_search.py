# faiss_search.py
import faiss
import numpy as np

def build_faiss_index(embedding_file):
    catalog_embeddings = np.load(embedding_file).astype('float32')
    embedding_dim = catalog_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(catalog_embeddings)
    return index

def search_similar_products(index, query_embedding, k=5):
    query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return distances, indices
