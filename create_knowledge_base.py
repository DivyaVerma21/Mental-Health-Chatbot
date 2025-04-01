import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load PDFs
def load_pdf_files(data_path):
    try:
        loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} pages from PDF files.")
        return documents
    except Exception as e:
        logging.error(f"Error loading PDFs: {e}")
        return []

documents = load_pdf_files(DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data, chunk_size=500, chunk_overlap=50):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = text_splitter.split_documents(extracted_data)
        logging.info(f"Created {len(text_chunks)} text chunks.")
        return text_chunks
    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        return []

text_chunks = create_chunks(documents)

# Step 3: Load Embedding Model
def get_embedding_model():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logging.info("Loaded embedding model successfully.")
        return embedding_model
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        return None

embedding_model = get_embedding_model()

# Step 4: Store Embeddings in FAISS
def save_embeddings_to_faiss(text_chunks, embedding_model, db_path):
    if os.path.exists(db_path):
        logging.warning(f"FAISS database already exists at {db_path}. Skipping creation.")
        return

    try:
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(db_path)
        logging.info(f"FAISS database saved at {db_path}.")
    except Exception as e:
        logging.error(f"Error creating FAISS database: {e}")

if embedding_model and text_chunks:
    save_embeddings_to_faiss(text_chunks, embedding_model, DB_FAISS_PATH)

