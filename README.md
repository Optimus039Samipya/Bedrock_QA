In my earlier project you will find that I have made a QA pipeline using AWS SageMaker and it turned out to be very costly, plus there was no instance strong enough in the free tier that could be utilised for the model training. Bedrock was a better alternative as the service charges you per token instead of the running time, and hence turned out to be more cost effective, not to mention that it was already allowing access to huge foundational models like Llama2 (70B).



🧠 RAG over PDF using AWS Bedrock + LLaMA3 + FAISS

This project implements a Retrieval-Augmented Generation (RAG) pipeline using:

- 📄 PDF document input (PyMuPDF)
- 🧩 Chunking via LangChain's RecursiveCharacterTextSplitter
- 🤖 Amazon Titan Embeddings (via AWS Bedrock)
- 🔍 FAISS for similarity search
- 🦙 Meta LLaMA3-70B Instruct (via AWS Bedrock) for question answering

🚀 Features

- Ask questions over any PDF using natural language.
- Embeds your PDF text using Amazon Titan.
- Retrieves relevant document chunks using FAISS.
- Generates answers with LLaMA3 hosted on Bedrock.

🛠️ Requirements

Make sure you have Python 3.8+ installed, then:

pip install boto3 faiss-cpu pymupdf numpy langchain

You also need:

✅ A valid AWS account with:
  - Bedrock access to:
    - amazon.titan-embed-text-v2:0
    - meta.llama3-70b-instruct-v1:0
  - Billing enabled (credit card attached)

📄 Usage

1. Put your PDF file in the project folder.
2. Edit the script to point to your file:
   pdf_path = "your_file.pdf"
3. Run the script:
   python your_script.py
4. Ask questions in the terminal. Example:
   Ask a question: What is normalization in databases?
5. To quit:
   Ask a question: q

📦 Project Structure

.
├── your_script.py         # Main RAG pipeline
├── 4th-Sem_CSE_Database-Management-System.pdf
└── README.txt

🧠 How it Works

PDF → Text Extraction → Chunking → Embedding via Titan → FAISS Index
                                           ↓
                                User Query (Embedding)
                                           ↓
                          FAISS Vector Search → Relevant Chunks
                                           ↓
                      Prompt → LLaMA3 → Final Answer Generated

🔐 AWS Credentials

Make sure your AWS credentials are set up in ~/.aws/credentials or via environment variables. Example:

export AWS_ACCESS_KEY_ID=your_key  
export AWS_SECRET_ACCESS_KEY=your_secret  
export AWS_REGION=your_region  

📢 Notes

- If you're getting "INVALID_PAYMENT_INSTRUMENT" or "access denied" errors, check your billing setup and model permissions in the AWS Bedrock console.
- Amazon Titan does not support batch embedding, so the script embeds each chunk one at a time.

