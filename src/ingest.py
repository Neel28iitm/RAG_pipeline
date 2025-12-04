import os
import glob
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "data"
CHROMA_PATH = "chroma_db"

def load_documents(files=None):
    if files:
        pdf_files = files
    else:
        # Search recursively for PDFs
        pdf_files = glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True)
    
    if not pdf_files:
        print("No PDF files found.")
        return []

    print(f"Found {len(pdf_files)} PDFs. Loading all of them...")

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks, chroma_path=CHROMA_PATH):
    # Note: We expect chroma_path to be a fresh directory provided by the caller
    # so we don't need to delete it here.

    print("Initialize Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    print(f"Saving {len(chunks)} chunks to ChromaDB at {chroma_path}...")
    
    # Initialize DB
    db = Chroma(
        embedding_function=embeddings, 
        persist_directory=chroma_path
    )

    # Batch processing to avoid rate limits
    batch_size = 5  # Small batch size to be safe
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  - Processing batch {i//batch_size + 1}/{total_batches}...")
        try:
            db.add_documents(batch)
            time.sleep(2) # Wait 2 seconds between batches
        except Exception as e:
            print(f"  Error saving batch: {e}")
        
    print(f"✅ Saved {len(chunks)} chunks to {chroma_path} successfully.")

def main():
    documents = load_documents()
    if not documents:
        print("No documents loaded.")
        return
        
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()