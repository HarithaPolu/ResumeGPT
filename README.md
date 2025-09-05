# ResumeGPT – Personal Knowledgebot (RAG Demo)

A lightweight **Retrieval-Augmented Generation (RAG)** chatbot that lets users ask questions about your **resume** and **recent projects**. Documents are chunked, embedded, stored in **ChromaDB**, and retrieved to ground answers from an LLM. Built with **Streamlit** + **LangChain**.

> 🎯 Perfect for portfolio demos, interviews, and showcasing practical RAG.

---

## ✨ Features

- 📄 Upload **PDF / TXT / Markdown** resume & project docs  
- ✂️ Smart chunking with overlap for better retrieval  
- 🔎 Vector search with **ChromaDB**  
- 🧠 Choice of embeddings:
  - **OpenAI** (paid/trial)
  - **Hugging Face** (`all-MiniLM-L6-v2`) – **free**
- 💬 Context-aware answers with LangChain `RetrievalQA`
- 🖥️ Clean **Streamlit** UI with chat history & sample questions
- 🧹 UTF-8 sanitization to avoid emoji/encoding crashes with vector stores

---

## 🧱 Architecture (RAG Pipeline)

1. **Ingest** – Load docs (PyPDFLoader/TextLoader) + metadata  
2. **Chunk** – RecursiveCharacterTextSplitter (size ~500, overlap ~50)  
3. **Embed** – Convert chunks to vectors (OpenAI or Hugging Face)  
4. **Store** – Save vectors + metadata in **ChromaDB** (persisted)  
5. **Retrieve** – Top-K similar chunks for each user query  
6. **Generate** – LLM answers with retrieved context (LangChain `RetrievalQA`)  
7. **UI** – Streamlit app for uploads, processing, and chat

---

## 📁 Project Structure

├─ ResumeGPT.py # Streamlit app
├─ requirements.txt
├─ .env # API keys (not committed)
├─ chroma_db/ # Persisted vector store (auto-created)
└─ README.md

---

## ⚙️ Requirements

- Python 3.10+
- Works with the modern LangChain split packages

**requirements.txt**
```txt
streamlit==1.28.1
langchain==0.1.17
langchain-openai==0.1.6
langchain-community==0.0.36
chromadb==0.4.15
pypdf==3.17.1
tiktoken==0.5.1
python-dotenv==1.0.0
sentence-transformers==2.2.2   # for free Hugging Face embeddings option
# Option A: OpenAI for embeddings + LLM
OPENAI_API_KEY=sk-...

# Option B: Hugging Face embeddings (free) – no key needed
# (App will still run if OPENAI_API_KEY is empty; see toggle in UI/code)
# 1) Create & activate venv (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Start the app
streamlit run ResumeGPT.py
Using the App

Enter your OpenAI API key in sidebar (or leave empty to use free Hugging Face embeddings if your code has that toggle).

Upload your resume and project documents (PDF/TXT/MD).

Click “Process Documents”.

Ask questions like:

“What are my key technical skills?”

“Summarize my work experience.”

“List my recent projects and tools used.”
Configuration (defaults)

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

TOP_K_RESULTS = 3

ChromaDB persist_directory = ./chroma_db
