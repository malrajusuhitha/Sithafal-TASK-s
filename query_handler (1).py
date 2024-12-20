from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


vector_store = FAISS.load_local("./vector_store")
embeddings = OpenAIEmbeddings()  # Configure parameters if needed.

# Load the FAISS vector store with embeddings
vector_store = FAISS.load_local("./vector_store", embeddings)
vector_store = FAISS.load_local("./embeddings.bin")

def query_vector_db(query, embedding_model):
    query_embedding = embedding_model.encode([query])
    results = vector_store.similarity_search(query_embedding, k=5)
    return results