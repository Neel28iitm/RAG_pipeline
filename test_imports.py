import sys
import os

# Add src to path to simulate streamlit behavior or running from src
sys.path.append(os.path.join(os.getcwd(), 'src'))

print("Attempting to import query...")
try:
    import query
    print("Successfully imported query.")
except Exception as e:
    print(f"Failed to import query: {e}")

print("Attempting to import langchain_chroma...")
try:
    from langchain_chroma import Chroma
    print("Successfully imported langchain_chroma.")
except Exception as e:
    print(f"Failed to import langchain_chroma: {e}")
