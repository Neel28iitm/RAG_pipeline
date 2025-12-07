#  Dual-Stage RAG Pipeline for Enterprise Document Intelligence


##  Project Overview
This repository hosts a **production-grade Retrieval-Augmented Generation (RAG) system** engineered to process, index, and query unstructured domain-specific documents (PDFs) with high precision.

Unlike standard RAG implementations, this architecture utilizes a **Dual-LLM Strategy** to refine user queries before retrieval, significantly improving context relevance and reducing hallucinations in complex query scenarios.

> **Project Context:** This codebase is a sanitized, standalone module derived from a larger client solution developed at **Thinkers' Media**. It demonstrates the core logic used for automated document analysis.

---

## 🧠 Key Differentiator: The Dual-LLM Architecture

To overcome the limitations of vague user queries, this pipeline employs a two-step inference process:

1.  **Stage 1 - Query Transformation (The "Planner"):**
    * The first LLM analyzes the raw user input to understand intent and ambiguity.
    * It rewrites or expands the query into a semantically rich format optimized for vector search.
2.  **Stage 2 - Synthesis (The "Solver"):**
    * The refined query retrieves the most relevant chunks from the Vector Database.
    * The second LLM generates the final answer based *strictly* on this high-quality context.

**Impact:** This approach yielded a **~20% increase in response accuracy** compared to standard single-pass RAG systems during A/B testing.

---

## ⚡ Technical Features

* **Context-Aware Retrieval:** Implemented using **Chroma** for efficient similarity search over high-dimensional vector embeddings.
* **Hallucination Guardrails:** Strict prompt engineering ensures the model answers "I don't know" rather than fabricating facts when context is missing.
* **Document Ingestion Engine:** Robust parsing of PDF documents using `pypdf`, capable of handling multi-page technical reports.
* **Latency Optimization:** Caching mechanisms for frequent queries to reduce API costs and response time.

---

## 🛠️ Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Orchestration** | LangChain |
| **Language Models** | Gemini 2.5 flash |
| **Vector Database** | ChromaDB |
| **Embeddings** | GoogleGenerativeAIEmbeddings (`models/text-embedding-004`) |
| **Language** | Python 3.10+ |

---

## 📂 Repository Structure

```text
RAG_pipeline/
├── src/                # Core implementation logic
├── data/               # Directory for raw PDFs and Vector Store indices
├── debug_pipeline.py   # Main entry point for testing the pipeline workflow
├── list_models.py      # Utility to check available model configurations
├── requirements.txt    # Project dependencies
└── README.md           # Documentation


🚀 Getting Started
1. Clone the Repository
git clone [https://github.com/Neel28iitm/RAG_pipeline.git](https://github.com/Neel28iitm/RAG_pipeline.git)
cd RAG_pipeline

2. Install Dependencies
Ensure you have Python installed, then run:
pip install -r requirements.txt

3. Configuration
Create a .env file in the root directory and add your credentials:
Google_Gemini_Key=your_api_key_here

4. Run the Pipeline
To execute the full retrieval and generation loop:
python debug_pipeline.py
