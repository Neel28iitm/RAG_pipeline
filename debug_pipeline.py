import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

def debug_pdf_extraction(pdf_path):
    print(f"\n--- Debugging Extraction for: {pdf_path} ---")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages.")
        
        # Print first 500 chars of first page to verify text quality
        if documents:
            print("\n[Preview of Page 1 Text]:")
            print("-" * 40)
            print(documents[0].page_content[:500])
            print("-" * 40)
            
            # Check for empty or sparse text
            total_text_len = sum(len(doc.page_content) for doc in documents)
            print(f"Total characters extracted: {total_text_len}")
            if total_text_len < 100:
                print("⚠️ WARNING: Very little text extracted. PDF might be scanned/image-based.")
        
        return documents
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return []

def debug_chunking(documents):
    print(f"\n--- Debugging Chunking ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    
    if chunks:
        print("\n[Preview of Chunk 1]:")
        print("-" * 40)
        print(chunks[0].page_content)
        print("-" * 40)
    
    return chunks

def debug_retrieval(chunks, query_text):
    print(f"\n--- Debugging Retrieval for query: '{query_text}' ---")
    
    # Create a temporary in-memory DB for debugging
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    print("Creating temporary vector store...")
    # Initialize empty Chroma
    db = Chroma(
        embedding_function=embeddings,
        collection_name="debug_collection"
    )
    
    # Add documents in batches
    import time
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  - Embedding batch {i//batch_size + 1}...")
        try:
            db.add_documents(batch)
            time.sleep(2)
        except Exception as e:
            print(f"  Error embedding batch: {e}")
    
    print("Searching...")
    results = db.similarity_search_with_score(query_text, k=3)
    
    print(f"\n[Top 3 Results]:")
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} (Score: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content Preview: {doc.page_content[:200]}...")
        
    # Cleanup
    db.delete_collection()

def main():
    # Find a PDF to test
    pdf_files = glob.glob("data/**/*.pdf", recursive=True)
    if not pdf_files:
        # Try to find temp files if no data files
        import tempfile
        temp_dir = tempfile.gettempdir()
        # This is a guess, we might need the user to point us to a file
        print("No PDFs found in 'data'. Please provide a path to a PDF.")
        return

    test_pdf = pdf_files[0]
    
    documents = debug_pdf_extraction(test_pdf)
    if documents:
        chunks = debug_chunking(documents)
        # Skip retrieval for now to avoid API errors and focus on text quality
        print("\n⚠️ Skipping retrieval to avoid API Rate Limits.")
        print("Check the text preview above. If it looks like garbage (e.g., random symbols, empty), that is the root cause.")

if __name__ == "__main__":
    main()
