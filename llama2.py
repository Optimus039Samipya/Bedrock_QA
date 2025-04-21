import json
import faiss
import fitz  # PyMuPDF
import boto3
import numpy as np
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

#AWS BEDROCK 
bedrock = boto3.client("bedrock-runtime")

# FITZ TO OPEN PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return full_text

#LANGCHAIN RECURSIVE TEXT SPLITTER FOR CHUNKING
def split_into_chunks(text: str, chunk_size=500, chunk_overlap=100) -> List[str]:
    # Wrap in Document class for LangChain
    documents = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"🔹 Total chunks created: {len(split_docs)}")
    return [doc.page_content for doc in split_docs]

#USING AMAZON TITAN'S EMBEDDING
def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        body = {
            "inputText": text  # 👈 Just one string at a time
        }

        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        embeddings.append(result["embedding"])  # Titan always returns {"embedding": [...]}
    return embeddings

# INDEXING USING FAISS
def build_faiss_index(text_chunks: List[str]):
    embeddings = get_embeddings(text_chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, text_chunks, np.array(embeddings).astype("float32")

#RETRIEVAL 
def retrieve_chunks(query: str, index, text_chunks: List[str], embeddings: np.ndarray, k=3) -> List[str]:
    query_embedding = get_embeddings([query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return [text_chunks[i] for i in I[0]]

# FEEDING INTO LLAMA2 70B MODEL
def generate_answer(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(f"- {chunk}" for chunk in context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
  
    payload = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    response = bedrock.invoke_model(
        modelId="meta.llama3-70b-instruct-v1:0",
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["generation"]

# --- Example Usage ---
if __name__ == "__main__":
    #Load PDF and split
    pdf_path = "4th-Sem_CSE_Database-Management-System.pdf"  
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(raw_text, chunk_size=500, chunk_overlap=100)

    # Embed + Index
    index, stored_chunks, stored_embeddings = build_faiss_index(chunks)

    # Ask a question

    
    while(True):
        query = input("Ask a question: PRESS Q,q TO EXIT\n")
        if query.lower()=="q":
            print("Exiting Bye!!!")
            break
        context_chunks = retrieve_chunks(query, index, stored_chunks, stored_embeddings)
        answer = generate_answer(query, context_chunks)

        print("\nRetrieved Context:")
        for c in context_chunks:
            print("-", c)

        print("\nAnswer:")
        print(answer)
