"""
Run RAG evaluation against test queries to understand retrieval quality.

This helps you learn:
- Which queries work well (good context retrieved)
- Which queries fail (need better query rewriting)
- How much context is typically needed
- Which documents are most relevant
"""

import asyncio
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from rag_evaluator import RAGEvaluator
from prompts import SYNTHESIS_PROMPT_TEMPLATE

# Setup
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("whitepapers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# Test queries to understand RAG
TEST_QUERIES = [
    "How does Apache Iceberg handle atomic transactions?",
    "What are the core benefits of Lakebase?",
    "How does Iceberg handle metadata and snapshots?",
    "What is the difference between Iceberg and traditional data formats?",
    "How do you migrate to Apache Iceberg?",
]

def evaluate_single_query(query: str, evaluator: RAGEvaluator) -> None:
    """Retrieve and evaluate one query."""
    print(f"\n📊 Evaluating: {query}")
    print("-" * 80)

    # Retrieve context
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)

    # Extract sources and context
    sources = []
    context_blocks = []
    for node in nodes:
        text = node.text
        if "Table of Contents" not in text and "--- image ---" not in text:
            context_blocks.append(text)
            source = node.metadata.get("source_file", "Unknown")
            sources.append(source)

    context = "\n\n".join(context_blocks)

    # Evaluate
    result = evaluator.evaluate_retrieval_quality(query, context, sources)

    # Display results
    print(f"✅ Context Length: {result['context_length']} chars")
    print(f"📄 Sources Found: {', '.join(set(sources))}")
    print(f"🎯 Judge Verdict: {result['judge_verdict']}")
    print(f"   → Sufficient: {result['is_sufficient']}")

    # Show snippet
    print(f"\n💡 Context Snippet:\n{context[:300]}...")

def main():
    evaluator = RAGEvaluator()

    print("🚀 RAG Evaluation Suite")
    print("=" * 80)
    print("Testing retrieval quality across benchmark queries...\n")

    for query in TEST_QUERIES:
        evaluate_single_query(query, evaluator)

    # Summary
    print("\n" + "=" * 80)
    print("📈 EVALUATION SUMMARY")
    print("=" * 80)

    summary = evaluator.summarize_evaluation()
    print(f"Total Queries Tested: {summary['total_queries']}")
    print(f"Sufficient Context Rate: {summary['sufficient_context_rate']:.1%}")
    print(f"Avg Context Length: {summary['avg_context_length']:.0f} chars")
    print(f"Avg Sources Per Query: {summary['avg_sources_per_query']:.1f}")

    # Save report
    evaluator.save_evaluation_report()

    print("\n💡 Key Learning: ")
    if summary['sufficient_context_rate'] > 0.8:
        print("✅ Your retrieval is excellent! Most queries get good context.")
    elif summary['sufficient_context_rate'] > 0.5:
        print("⚠️  Your retrieval is okay but could improve with better query rewriting.")
    else:
        print("❌ Your retrieval needs work. Try: larger top_k, better embeddings, or smarter prompts.")

if __name__ == "__main__":
    main()
