
#!pip install -q gpt4all sentence-transformers faiss-cpu numpy PyMuPDF
#!pip install gradio

import fitz 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gpt4all import GPT4All
import gradio as gr
import os
import time
import requests
path="/home/abhishekyadav/stig_project/src/model_download/"
import os
import torch
import faiss
import requests


models = GPT4All.list_models()
for m in models:
    print(m["filename"])

# Load sentence transformer model

model_name = "all-MiniLM-L6-v2"
local_path=f"{path}{model_name}"

if os.path.exists(local_path):
    print("Loading local model...")
    embedder = SentenceTransformer(local_path)

else:
    print("Model not found. Downloading...")
    embedder = SentenceTransformer(model_name)
    embedder.save(local_path)



def generate(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,

                "options": {
                        "num_predict": 200,
                        "temperature": 0.2,
                        "top_p": 0.9
                            }
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["response"]
        
        else:
            print("API error:", response.status_code)
            return "Error: LLM API not working"

    except requests.exceptions.ConnectionError:
        print("Ollama API is not running")
        return "Error: Ollama not running"

    except Exception as e:
        print("Unexpected error:", str(e))
        return f"Error: {str(e)}"
    

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    document = fitz.open(pdf_path)
    pdf_text = ""

    # Loop through all pages and extract text
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pdf_text += page.get_text("text")  

    return pdf_text

def build_chunks(text, chunk_size=200, overlap=40):
    words = text.split()

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

cache = {}

def rag_query(query, top_k=2):
    global index, documents, pdf_loaded

    start_total = time.time()
    # STEP 1: Check cache first
    key = query.strip().lower()
    if key in cache:
        print("From cache")
        return cache[key]

    if index is None or documents is None or len(documents) == 0 or pdf_loaded is False:
      return "Please upload a PDF first!"

    # Embed the query
    t1 = time.time()
    query_vec = embedder.encode([query])
    print("Embedding time:", time.time() - t1)

    # Retrieve top-k docs
    t2 = time.time()
    distances, indices = index.search(np.array(query_vec), top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    print("Retrieval time:", time.time() - t2)

    # Create context
    t3 = time.time()
    context = "\n".join(retrieved_docs)
    print("Context build time:", time.time() - t3)

    # Prompt
    prompt =  f"""
            You are a helpful assistant. Answer the question using ONLY the provided context.

            If the answer is not explicitly present in the context, respond exactly with:
            "not found in pdf file"

            Context:
            {context}

            Question:
            {query}

            Answer:
                    """
    
    t4 = time.time()

    response = generate(prompt)
    print("LLM time:", time.time() - t4)
    print("TOTAL TIME:", time.time() - start_total)
    print("________________________________________")

    # clean output
    answer = response.strip().split("\n")[0]

    #  STEP 2: Store in cache
    cache[key] = answer

    return answer

pdf_loaded = False
def load_pdf(pdf_file):
    global documents, index, pdf_loaded

    # check if file is uploaded
    if pdf_file is None:
        return "Please upload a file first before clicking Load"

    file_path = pdf_file.name
    ext = os.path.splitext(file_path)[1].lower()

    # ---------- PDF ----------
    if ext == ".pdf":
        text = extract_pdf_text(file_path)

    # ---------- TXT ----------
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    else:
        return "Unsupported file type"


    text = extract_pdf_text(pdf_file)
    documents = build_chunks(text)

    embeddings = embedder.encode(documents)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    pdf_loaded = True
    return f"✅ File loaded successfully ({ext}, {len(documents)} chunks)"



# Sample questions add in 
question_list = [
    "What is the purpose of FAISS ?",
    "Where is HCL headquartered ?",
    "How many employees does HCL have ?",
    "What services does HCL provide ?",
    "How does HCL promote innovation ?"
]


# RAG wrapper
def ask_question(q):

    if not pdf_loaded:
      return "Please upload a file first!"

    return rag_query(q)


# ---------- UI ----------
with gr.Blocks() as app:

    gr.Markdown("#PDF RAG  with GPT4All Chatbot")

    # PDF upload
    file_input = gr.File(label="Upload PDF or TXT", file_types=[".pdf", ".txt"])
    load_btn = gr.Button("Load File")
    status = gr.Textbox(label="Status")

    load_btn.click(load_pdf, inputs=file_input, outputs=status)

    # ---------- Dropdown ----------
    gr.Markdown("##  Select a Question")

    question_dropdown = gr.Dropdown(
        choices=question_list,
        label="Choose a question",
        value=question_list[0]  # default selected 
    )

    ask_btn = gr.Button("Get Answer")
    answer = gr.Textbox(label="Answer", lines=9,
    max_lines=20,
    interactive=False)

    ask_btn.click(ask_question, inputs=question_dropdown, outputs=answer)

    # ---------- Free text ----------
    gr.Markdown("## Or Ask Your Own Question")

    custom_q = gr.Textbox(label="Type your question")
    custom_btn = gr.Button("Ask")

    custom_btn.click(ask_question, inputs=custom_q, outputs=answer)

app.launch(share=True)