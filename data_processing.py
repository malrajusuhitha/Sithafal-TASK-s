import fitz  # PDF processing library (PyMuPDF)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# 1. Extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text

# 2. Extract text from multiple PDFs
def extract_text_from_pdfs(pdf_folder):
    """
    Extracts text from all PDF files in a folder.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with 'file_name' and 'text'.
    """
    pdf_texts = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts.append({"file_name": file_name, "text": text})
    return pdf_texts

# 3. Segment text into chunks
def segment_text(text, chunk_size=256):
    """
    Splits text into smaller chunks.

    Args:
        text (str): Text to split.
        chunk_size (int): Maximum number of characters per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 4. Embed text chunks
def embed_chunks(model_name, tokenizer, chunks):
    """
    Generates embeddings for chunks using a sequence-to-sequence model.

    Args:
        model_name (str): Name of the model to load.
        tokenizer: Tokenizer object.
        chunks (List[str]): List of text chunks.

    Returns:
        List[torch.Tensor]: List of embeddings for each chunk.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    embeddings = []

    for chunk in chunks:
        # Tokenize the chunk
        inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True, padding=True)

        # Pass inputs through the model
        output = model(**inputs)

        # Extract the encoder's last hidden states
        sentence_embedding = output.encoder_last_hidden_state.mean(dim=1)  # Mean pooling over sequence
        embeddings.append(sentence_embedding)

    return embeddings

# 5. Store embeddings and chunks
def store_embeddings(file_name, chunks, embeddings, storage_path):
    """
    Stores embeddings and chunks for efficient retrieval.

    Args:
        file_name (str): Name of the PDF file.
        chunks (List[str]): Text chunks.
        embeddings (List[torch.Tensor]): Chunk embeddings.
        storage_path (str): Path to save the embeddings and chunks.
    """
    os.makedirs(storage_path, exist_ok=True)
    torch.save({"file_name": file_name, "chunks": chunks, "embeddings": embeddings}, 
               os.path.join(storage_path, f"{file_name}_data.pt"))

# Main pipeline
if __name__ == "__main__":
    # Define model and tokenizer
    model_name = "facebook/bart-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define paths
    pdf_folder = "./pdfs"  # Folder containing the PDF files
    storage_path = "./embeddings"  # Folder to store embeddings

    # Ensure the PDF folder exists
    if not os.path.exists(pdf_folder):
        print(f"Error: The specified PDF folder '{pdf_folder}' does not exist.")
        os.makedirs(pdf_folder)
        print(f"Created an empty directory at '{pdf_folder}'. Please add PDF files to this directory and rerun the script.")
        exit()

    # Check if the folder contains files
    if not os.listdir(pdf_folder):
        print(f"The directory '{pdf_folder}' is empty. Add some PDF files and rerun the script.")
        exit()

    # Process PDFs
    pdf_texts = extract_text_from_pdfs(pdf_folder)
    for pdf in pdf_texts:
        chunks = segment_text(pdf["text"])  # Segment the extracted text into chunks
        embeddings = embed_chunks(model_name, tokenizer, chunks)  # Generate embeddings
        store_embeddings(pdf["file_name"], chunks, embeddings, storage_path)  # Store chunks and embeddings

    print("Processing completed.")
