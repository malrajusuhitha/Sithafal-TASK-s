from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    return model.encode(chunks, convert_to_tensor=True)