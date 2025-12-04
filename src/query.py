import os
import argparse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# if not GOOGLE_API_KEY:
#     print("Error: GOOGLE_API_KEY not found in environment variables.")
#     exit(1)

PROMPT_TEMPLATE = """
You are an expert AI assistant tasked with answering questions based on the provided document context.

Context:
{context}

---

Question: {question}

Instructions:
1. Answer the question in a comprehensive and detailed manner.
2. If the answer involves a list or steps, use bullet points for better readability.
3. Use professional language suitable for a business or technical context.
4. Elaborate on key details found in the context (e.g., dates, specific technologies, metrics).
5. If the answer is not explicitly found in the context, state "I cannot find this information in the document" but do not make up facts.

Answer:
"""

def generate_search_query(original_query):
    """
    Generates an optimized search query using the LLM to improve retrieval.
    """
    print(f"\n--- Generating Search Query for: '{original_query}' ---")
    
    query_gen_prompt = """You are an expert at converting user questions into database search queries. 
    Your goal is to generate a search query that will retrieve the most relevant documents for the user's question.
    
    User Question: {question}
    
    Instructions:
    1. Identify the core intent and keywords from the user's question.
    2. Expand on the keywords with synonyms or related terms if necessary to broaden the search coverage.
    3. Remove conversational filler words (e.g., "please", "tell me about", "I want to know").
    4. Return ONLY the optimized search query text. Do not include any explanations or quotes.
    
    Optimized Search Query:"""
    
    prompt_template = ChatPromptTemplate.from_template(query_gen_prompt)
    prompt = prompt_template.format(question=original_query)
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    response = model.invoke(prompt)
    
    optimized_query = response.content.strip()
    print(f"Original Query: {original_query}")
    print(f"Optimized Query: {optimized_query}")
    return optimized_query

def query_rag(query_text, chroma_path=CHROMA_PATH):
    # Prepare the DB.
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # 1. Generate Optimized Search Query
    search_query = generate_search_query(query_text)

    # Search the DB.
    # Note: Chroma's default distance is L2 or Cosine distance where lower is better/different.
    # However, LangChain's wrapper might return different scores.
    # We'll use similarity_search_with_score which returns (doc, score).
    results = db.similarity_search_with_score(search_query, k=10)
    
    # Debug: Print results to console
    print(f"\n--- Query: {search_query} ---")
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1}: Score={score:.4f}, Source={doc.metadata.get('source')}")

    if len(results) == 0:
        print("No results found.")
        return "Unable to find matching results.", [], ""

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Use the ORIGINAL query for the final answer generation to ensure we answer what the user asked
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    response = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return response.content, sources, context_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    response_text, sources, _ = query_rag(query_text)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
