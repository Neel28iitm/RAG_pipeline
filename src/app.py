import streamlit as st
import os
import tempfile
import ingest
import query

# Page Config
st.set_page_config(page_title="Chat with PDF", page_icon="📄")

st.title("Chat with PDF 📄")

# Sidebar for File Upload
with st.sidebar:
    st.header("Configuration")
    
    # API Key Handling
    api_key = st.text_input("Google API Key", type="password", help="Enter your Google API Key here if not set in .env")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ Please enter your Google API Key to proceed.")
        st.stop()

    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary directory to preserve filename
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        if st.button("Process Document"):
            with st.spinner("Processing..."):
                try:
                    # Purani chat clear kar do
                    st.session_state.messages = []
                    
                    # Ingest the document
                    documents = ingest.load_documents([tmp_file_path])
                    if documents:
                        chunks = ingest.split_text(documents)
                        # Create a unique path for this document's DB
                        new_db_path = tempfile.mkdtemp()
                        st.session_state.chroma_path = new_db_path
                        
                        ingest.save_to_chroma(chunks, chroma_path=new_db_path)
                        st.success("Ingestion Complete! You can now chat with the document.")
                    else:
                        st.error("Could not load documents.")
                except Exception as e:
                    st.error(f"Error during processing: {e}")
                finally:
                    # Clean up temp file
                    # if os.path.exists(tmp_file_path):
                    #     os.remove(tmp_file_path)
                    pass

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chroma_path" not in st.session_state:
    st.session_state.chroma_path = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the PDF..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.chroma_path:
                    response_text, sources, context_text = query.query_rag(prompt, chroma_path=st.session_state.chroma_path)
                else:
                    st.error("Please upload and process a document first.")
                    response_text = "No document loaded."
                    sources = []
                    context_text = ""
                
                # Format response with sources
                full_response = response_text
                if sources:
                    unique_sources = list(set(sources))
                    # Clean up source paths for display (optional)
                    unique_sources = [os.path.basename(s) for s in unique_sources] 
                    full_response += f"\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in unique_sources])
                
                st.markdown(full_response)
                
                # Show retrieved context for debugging
                with st.expander("View Retrieved Context (Debug)"):
                    st.text(context_text)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
