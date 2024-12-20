from scraper import scrape_website, process_and_segment
from embedder import generate_embeddings
from query_handler import query_vector_db
from response_generator import generate_response
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # Assuming OpenAI embedding
import os


URLS = ["https://www.nmrec.edu.in/"]

if __name__ == "__main__":
    # Step 1: Scrape and Process Data
    all_chunks = []
    for url in URLS:
        content = scrape_website(url)
        chunks = process_and_segment(content)
        all_chunks.extend(chunks)

    # Step 2: Generate Embeddings
    embeddings = generate_embeddings(all_chunks)

    # Step 3: Store Embeddings in Vector Database
    # Use FAISS as an example
    from langchain.vectorstores import FAISS
    vector_store = FAISS.from_embeddings(embeddings, all_chunks)
    vector_store.save_local("./vector_store")

    # Step 4: Query Handling
    user_query = input("Enter your query: ")
    results = query_vector_db(user_query, embedding_model) # type: ignore

    # Step 5: Generate Response
    context = "\n".join([result.text for result in results])
    response = generate_response(user_query, context)
    print(response)