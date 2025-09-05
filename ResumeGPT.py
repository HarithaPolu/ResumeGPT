# ResumeGPT - Personal RAG Chatbot (Optimized for Embeddings)
import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Still need this for the LLM
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings # This will be our primary for documents

# Load environment variables
load_dotenv()

# Config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
VECTORSTORE_PATH = "./chroma_db"

class ResumeGPT:
    def __init__(self):
        """Initialize RAG system"""
        # Use HuggingFaceEmbeddings as the default for document embeddings
        self.doc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.qa_embeddings = None # This will be set if an OpenAI key is provided and used for QA
        self.vectorstore = None
        self.qa_chain = None
        self.documents_loaded = False
        self.openai_api_key_set = False # Track if OpenAI key is successfully set

    def setup_openai(self, api_key: str):
        """Setup OpenAI API key for LLM and optionally for QA embeddings"""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
            self.qa_embeddings = OpenAIEmbeddings(openai_api_key=api_key) # Use OpenAI for QA embeddings if key is present
            self.openai_api_key_set = True
            st.session_state.openai_key_valid = True # Update session state for validity
        else:
            self.openai_api_key_set = False
            st.session_state.openai_key_valid = False

    def load_documents(self, uploaded_files) -> List:
        """Load PDF/TXT/MD files"""
        documents = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                elif uploaded_file.name.endswith(('.txt', '.md')):
                    loader = TextLoader(tmp_path, encoding='utf-8')
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
            finally:
                os.unlink(tmp_path)

        return documents

    def chunk_documents(self, documents) -> List:
        """Split documents into semantic chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            # Ensure metadata and page_content are properly encoded
            chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8")
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if isinstance(v, str):
                        chunk.metadata[k] = v.encode("utf-8", "ignore").decode("utf-8")
        return chunks

    def create_vectorstore(self, chunks, persist=True):
        """Create or load Chroma vectorstore using HuggingFaceEmbeddings"""
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.doc_embeddings, # Always use HuggingFaceEmbeddings for document ingestion
                persist_directory=VECTORSTORE_PATH if persist else None
            )
            if persist:
                self.vectorstore.persist()
            st.success(f"✅ Vectorstore ready with {len(chunks)} chunks using HuggingFaceEmbeddings!")
            return True
        except Exception as e:
            st.error(f"Error creating vectorstore: {e}")
            return False

    def setup_qa_chain(self, api_key: str):
        """Setup RetrievalQA chain. Requires an OpenAI API key for ChatOpenAI."""
        if not self.openai_api_key_set:
            st.error("OpenAI API key is required for the Language Model (LLM) to generate answers.")
            return False

        prompt = """
        You are a helpful AI assistant answering questions about a person's resume and projects.
        Use the context below. If info is missing, say "I don't have that information."

        Context:
        {context}

        Question: {question}

        Answer:"""
        PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])

        # The retriever uses the embedding model that Chroma was initialized with (HuggingFaceEmbeddings)
        # for similarity search against the user's query.
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS}
        )

        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=0.7,
                    max_tokens=500,
                    openai_api_key=api_key # This still requires the OpenAI key for the LLM
                ),
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            self.documents_loaded = True
            return True
        except Exception as e:
            st.error(f"Error setting up QA chain. Ensure your OpenAI API key is valid: {e}")
            self.documents_loaded = False
            return False

    def query(self, question: str) -> Dict[str, Any]:
        """Query RAG system"""
        if not self.documents_loaded:
            return {"error": "No documents loaded or QA chain not set up. Please upload and process documents with a valid OpenAI key."}
        if not self.openai_api_key_set:
            return {"error": "OpenAI API key is required for the Language Model to generate answers."}
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            return {"error": f"Query failed: {e}"}

# Streamlit app
def main():
    st.set_page_config(page_title="ResumeGPT", page_icon="🤖", layout="wide")
    st.title("🤖 ResumeGPT - Personal Knowledge Chatbot")
    st.subheader("Ask questions about your resume and projects")

    if 'openai_key_valid' not in st.session_state:
        st.session_state.openai_key_valid = False

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key (Required for LLM)", type="password")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            st.success("✅ OpenAI API Key provided.")
            st.session_state.openai_key_valid = True
        else:
            st.warning("⚠️ Provide OpenAI API key or set in .env for LLM interaction.")
            st.session_state.openai_key_valid = False

        st.markdown("---")
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload resume/project documents",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True
        )

        st.markdown("---")
        st.header("🔧 RAG Parameters")
        st.info(f"Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}, Top-K: {TOP_K_RESULTS}")
        st.info("Document Embeddings: HuggingFace (all-MiniLM-L6-v2)")


    if 'resume_gpt' not in st.session_state:
        st.session_state.resume_gpt = ResumeGPT()
        st.session_state.documents_processed = False
        st.session_state.chat_history = []

    col1, col2 = st.columns([2, 1])
    with col1:
        if uploaded_files and not st.session_state.documents_processed:
            if st.button("🚀 Process Documents"):
                with st.spinner("Processing documents with HuggingFace Embeddings..."):
                    st.session_state.resume_gpt.setup_openai(api_key) # Attempt to setup OpenAI (for LLM)
                    documents = st.session_state.resume_gpt.load_documents(uploaded_files)
                    if documents:
                        chunks = st.session_state.resume_gpt.chunk_documents(documents)
                        if st.session_state.resume_gpt.create_vectorstore(chunks): # This uses HF Embeddings
                            # Only try to setup QA chain if an OpenAI key is valid
                            if st.session_state.openai_key_valid:
                                if st.session_state.resume_gpt.setup_qa_chain(api_key):
                                    st.session_state.documents_processed = True
                                    st.success("🎉 Documents processed! You can ask questions now.")
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to setup QA chain. Check OpenAI API key.")
                            else:
                                st.warning("Documents processed, but OpenAI API key needed for answering questions.")
                    else:
                        st.error("No documents loaded for processing.")


        if st.session_state.documents_processed and st.session_state.openai_key_valid:
            st.markdown("---")
            st.header("💬 Ask Questions")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {q[:60]}...", expanded=(i==len(st.session_state.chat_history)-1)):
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer:** {a}")

            question = st.text_input("Ask a question:", placeholder="e.g., What are my key skills?")
            if st.button("Ask Question") and question:
                with st.spinner("Thinking..."):
                    result = st.session_state.resume_gpt.query(question)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state.chat_history.append((question, result["answer"]))
                        st.experimental_rerun()
        elif st.session_state.documents_processed and not st.session_state.openai_key_valid:
            st.warning("Documents are processed, but an OpenAI API key is still required in the sidebar to ask questions (for the LLM).")
        elif uploaded_files and not api_key:
            st.warning("⚠️ Provide OpenAI API key to process documents and enable questioning.")
        elif api_key and not uploaded_files:
            st.info("📄 Upload documents to start")
        else:
            st.markdown("## Welcome to ResumeGPT! Upload docs → Process → Ask questions.")

    with col2:
        st.header("📊 System Status")
        st.success(f"🔑 OpenAI Key: {'Loaded' if st.session_state.openai_key_valid else 'Missing/Invalid'}")
        st.success(f"📄 Documents: {len(uploaded_files) if uploaded_files else 0}")
        st.success(f"🤖 RAG System: {'Ready (LLM enabled)' if st.session_state.documents_processed and st.session_state.openai_key_valid else 'Not Ready (LLM disabled)'}")

if __name__ == "__main__":
    main()
