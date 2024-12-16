Here's the complete documentation for your **Retrieval-Augmented Generation (RAG) pipeline project**, which processes PDF files, extracts text, segments, embeds, stores the data for efficient retrieval, and generates responses to user queries.

---

## **Retrieval-Augmented Generation (RAG) Pipeline Documentation**

### **Overview**

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** that interacts with semi-structured data stored in multiple PDF files. The system performs the following tasks:
1. Extracts text from PDFs.
2. Segments text into smaller chunks.
3. Embeds the chunks using a transformer model.
4. Stores the embeddings and text chunks for efficient retrieval.
5. Allows for querying the stored chunks and generates responses using a large language model (LLM).

### **Table of Contents**
1. [Setup and Installation](#setup-and-installation)
2. [PDF Processing](#pdf-processing)
3. [Text Embedding](#text-embedding)
4. [Chunking](#chunking)
5. [Retrieval](#retrieval)
6. [Query Processing and Response Generation](#query-processing-and-response-generation)
7. [How to Run the Project](#how-to-run-the-project)
8. [File Structure](#file-structure)
9. [References](#references)

---

### **1. Setup and Installation**

**Dependencies:**
- Python 3.7+
- `transformers` (for working with pre-trained models)
- `torch` (for PyTorch-based operations)
- `PyMuPDF` (for PDF text extraction)
- `scikit-learn` (for cosine similarity)
- `os`, `shutil`, and other standard libraries

**Installation:**
1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install required packages:
   ```bash
   pip install torch transformers pymupdf scikit-learn
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/retrieval-augmented-generation.git
   cd retrieval-augmented-generation
   ```

---

### **2. PDF Processing**

This component handles extracting text from PDF files using `PyMuPDF`. It supports processing both single and multiple PDFs.

#### **Extract Text from a Single PDF**

```python
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text from the PDF.
    """
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
```

#### **Extract Text from Multiple PDFs**

```python
def extract_text_from_pdfs(pdf_folder):
    """
    Extracts text from all PDF files in a folder.
    Args:
        pdf_folder (str): Path to the folder containing PDF files.
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing 'file_name' and 'text'.
    """
    pdf_texts = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts.append({"file_name": file_name, "text": text})
    return pdf_texts
```

---

### **3. Text Embedding**

Embeddings are generated for text chunks using a transformer-based sequence-to-sequence model like BART or T5.

#### **Embedding Function**

```python
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
```

---

### **4. Chunking**

Text is split into smaller chunks to improve memory efficiency and retrieval accuracy.

#### **Chunking Function**

```python
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
```

---

### **5. Retrieval**

This component retrieves relevant chunks based on a user query by comparing embeddings using cosine similarity.

#### **Cosine Similarity Function**

```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_chunks(query, model_name, tokenizer, storage_path):
    """
    Retrieves relevant chunks based on a query.
    Args:
        query (str): User query.
        model_name (str): Name of the model to load.
        tokenizer: Tokenizer object.
        storage_path (str): Path where embeddings and chunks are stored.
    Returns:
        List[str]: List of most relevant text chunks.
    """
    # Tokenize and embed the query
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    query_inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding=True)
    query_output = model(**query_inputs)
    query_embedding = query_output.encoder_last_hidden_state.mean(dim=1)

    relevant_chunks = []
    for file in os.listdir(storage_path):
        if file.endswith("_data.pt"):
            data = torch.load(os.path.join(storage_path, file))
            chunks = data["chunks"]
            embeddings = data["embeddings"]
            similarities = cosine_similarity(query_embedding.detach().numpy(), torch.stack(embeddings).detach().numpy())
            top_chunk_idx = similarities.argmax()
            relevant_chunks.append(chunks[top_chunk_idx])

    return relevant_chunks
```

---

### **6. Query Processing and Response Generation**

Once relevant chunks are retrieved, they are passed to a language model to generate a response.

#### **Response Generation**

```python
def generate_response(query, relevant_chunks, model_name, tokenizer):
    """
    Generates a response using the selected language model.
    Args:
        query (str): User query.
        relevant_chunks (List[str]): Relevant text chunks.
        model_name (str): Name of the model to load.
        tokenizer: Tokenizer object.
    Returns:
        str: Generated response.
    """
    context = " ".join(relevant_chunks)  # Combine chunks for context
    input_text = f"Question: {query} Context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

### **7. How to Run the Project**

1. **Prepare your environment:**
   - Set up a virtual environment and install dependencies.
   
2. **Place your PDF files in the `pdfs/` folder.**
   - Create the folder `pdfs/` in your project directory and add the PDF files you want to process.

3. **Run the pipeline:**
   - After setting up the PDF folder and ensuring the models are properly downloaded, run the script:
   ```bash
   python data_processing.py
   ```

4. **Query the model:**
   - Use the `retrieve_relevant_chunks()` and `generate_response()` functions to process queries and get answers based on the stored embeddings.

---

### **8. File Structure**

```
rag_pdf_chat/
├── pdfs/                  # Folder containing your PDF files
├── embeddings/            # Folder to store embeddings and chunk data
├── data_processing.py     # Main processing script
└── README.md              # Documentation
```

---

### **9. References**
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

This documentation provides an overview of the RAG pipeline, setup instructions, and guides on using the script to process PDFs and generate responses to queries. Feel free to extend or modify the pipeline based on your use case!


         
