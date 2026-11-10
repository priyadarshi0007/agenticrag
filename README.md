# Local Agentic RAG System

A learning-focused RAG system that works completely locally—no external APIs required.

## What It Does

Takes technical PDFs and answers questions by:
1. **Retrieving** relevant chunks from your documents
2. **Judging** if the context is sufficient
3. **Rewriting** the query if needed (up to 2 attempts)
4. **Synthesizing** a final answer grounded in the documents

## Architecture

```
PDF Ingestion (1_ingest_data.py)
    ↓
ChromaDB Vector Store (incremental, hash-based)
    ↓
Agentic Workflow (3_agentic_workflow.py)
    ├─ Retrieve: similarity search
    ├─ Judge: "enough context?"
    ├─ Rewrite: improve query if needed
    └─ Synthesize: generate answer
    ↓
Streamlit UI (app.py)
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start Ollama locally
ollama run gemma:2b
```

## Usage

### Option 1: Interactive Web UI
```bash
streamlit run app.py
```

### Option 2: Command Line
```bash
python 3_agentic_workflow.py --query "How does Iceberg handle transactions?"
```

### Option 3: Evaluate Retrieval Quality
```bash
python evaluate_rag.py
```

This runs benchmark queries and shows you:
- How much context each query retrieves
- Whether the judge thinks it's sufficient
- Which documents are found
- Aggregate success rate

## Add New PDFs

1. Place PDFs in `.data/` folder
2. Run: `python 1_ingest_data.py`
3. System automatically detects new files, parses them with Docling, and indexes vectors
4. Hashing prevents re-processing unchanged documents

## Key Concepts (for Learning)

**Query Decomposition**: If you ask "How do A and B compare?", the system splits it into:
- "How does A work?"
- "How does B work?"
- Then retrieves for both separately

**Judge Loop**: After retrieval, LLM judges: "Do I have enough info to answer?" 
- PASS → synthesize answer
- FAIL → rewrite query, try again (max 2 times)

**Incremental Ingestion**: PDFs are hashed. Only new/modified documents are re-processed.

## Evaluate & Learn

Run `evaluate_rag.py` to understand:
- Which queries retrieve good context (judge says PASS)
- Which queries fail (judge says FAIL)
- How many source documents are typically needed
- What context length is optimal

Use this to experiment:
- Change `similarity_top_k` in retrieval (3 vs 5 vs 10)
- Modify judge/rewriter prompts in `prompts.py`
- Try different embedding models
- Add more test queries

## Tech Stack

- **LLM**: Ollama (Gemma 2b, local)
- **Embeddings**: HuggingFace (BAAI/bge-small-en-v1.5, local)
- **Vector Store**: ChromaDB (persistent, local)
- **PDF Parsing**: Docling (smart tables, OCR-optional)
- **Workflows**: LlamaIndex Workflows (stateful, async)
- **UI**: Streamlit (interactive, real-time)

## Files

- `1_ingest_data.py` — Parse PDFs with Docling, embed with HuggingFace, store in ChromaDB
- `2_verify_db.py` — Quick test query to validate the vector store
- `3_agentic_workflow.py` — Core RAG logic (retrieve → judge → rewrite → synthesize)
- `app.py` — Streamlit UI with pipeline visualization
- `prompts.py` — Judge, rewriter, synthesis prompts
- `rag_evaluator.py` — Metrics to measure retrieval quality
- `evaluate_rag.py` — Benchmark suite to test your RAG

## Next Steps for Learning

1. Run `evaluate_rag.py` to see baseline retrieval quality
2. Experiment with different `similarity_top_k` values
3. Modify prompts in `prompts.py` and re-test
4. Try different embedding models in config
5. Add custom test queries and track success rate

---

Built to understand how RAG actually works under the hood. Not production-ready yet, but great for learning!
