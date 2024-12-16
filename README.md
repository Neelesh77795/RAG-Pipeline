 pip install pdfplumber
 pip install PyMuPDF sentence-transformers faiss-cpu openai textwrap
pip install sentence-transformers
import sentence_transformers
print(sentence_transformers.__version__)
pip install faiss-cpu
pip install faiss-gpu
pip install --upgrade pip
import faiss
print(faiss.__version__)
pip install openai
pip install --upgrade pip
import openai

openai.api_key = "your-openai-api-key"

   *** THOSE ARE THE REQURIMENTS ***

-------------------------------------------------------------
   import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
----------------------------------------------------------------
import textwrap

def chunk_text(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)
----------------------------------------------------------------
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(chunks):
    return [model.encode(chunk) for chunk in chunks]
---------------------------------------------------------------
import faiss
import numpy as np

def store_embeddings_in_faiss(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype(np.float32))
    return index
---------------------------------------------------------------
def query_faiss(index, query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]).astype(np.float32), k)
    return I[0]
------------------------------------------------------------------
import openai

openai.api_key = "your-openai-api-key"

def generate_response(query, context):
    prompt = f"Answer the following question using the context: {context}\nQuestion: {query}"
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt, 
        max_tokens=150
    )
    return response.choices[0].text.strip()
---------------------------------------------------------------------
def generate_comparison_response(query, chunks):
    # Extract relevant terms for comparison (customize this based on your data).
    comparison_data = []
    for chunk in chunks:
        comparison_data.append(extract_comparison_data(chunk))  # Custom function to extract comparison data.

    # Format comparison data as a table or bullet points.
    return format_comparison_as_table(comparison_data)  # Custom function to format the response.
-----------------------------------------------------------------------------
ALL TOGETHER 
def main(pdf_path, query):
    # Step 1: Extract and process PDF text
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    # Step 2: Embed the chunks and store in FAISS
    embeddings = embed_text(chunks)
    index = store_embeddings_in_faiss(embeddings)
    # Step 3: Embed the query and search for relevant chunks
    query_embedding = model.encode(query)
    relevant_chunk_indices = query_faiss(index, query_embedding)
     # Step 4: Retrieve the context and generate a response
    relevant_chunks = [chunks[i] for i in relevant_chunk_indices]
    context = " ".join(relevant_chunks)  # Combine relevant chunks for context
    response = generate_response(query, context)
    return response
