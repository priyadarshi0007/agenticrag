import os
import asyncio
import argparse
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.workflow import Event, StartEvent, StopEvent, step, Workflow
import chromadb
from prompts import JUDGE_PROMPT_TEMPLATE, REWRITER_PROMPT_TEMPLATE, SYNTHESIS_PROMPT_TEMPLATE

# 1. Configure Local M1 Pro Stack
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0)

# 2. Connect to Local ChromaDB Vector Store
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("whitepapers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# 3. Define State Events
class RetrievalEvent(Event):
    query: str
    context: str
    sources: list
    attempt: int

class EvaluationEvent(Event):
    query: str
    context: str
    sources: list
    evaluation: str
    attempt: int

# 4. Build the Stateful Agentic Workflow
class AgenticAnalystWorkflow(Workflow):
    
    @step
    async def retrieve_context(self, ev: StartEvent | EvaluationEvent) -> RetrievalEvent:
        if isinstance(ev, StartEvent):
            query = ev.get("query")
            attempt = 1
        else:
            query = ev.query
            attempt = ev.attempts + 1

        print(f"Searching local whitepapers for: '{query}'")
        
        # Keep top_k low and focused to prevent noise for gemma:2b
        retriever = index.as_retriever(similarity_top_k=3)
        
        # If the query asks about multiple concepts, split it into sub-queries
        queries = [query]
        if " and " in query.lower() or " vs " in query.lower():
            parts = query.replace(" and ", " vs ").split(" vs ")
            queries = [p.strip() for p in parts if len(p.strip()) > 3]
            print(f"🔍 Decomposed compound query into sub-queries: {queries}")

        nodes = []
        for q in queries:
            sub_nodes = retriever.retrieve(q)
            nodes.extend(sub_nodes)
        
        context_blocks = []
        sources = set()
        for node in nodes:
            text = node.text
            if "Table of Contents" in text or "--- image ---" in text:
                continue
            context_blocks.append(text)
            source_file = node.metadata.get("source_file", "Unknown PDF")
            sources.add(source_file)
            
        context = "\n\n".join(context_blocks)
        return RetrievalEvent(query=query, context=context, sources=list(sources), attempt=attempt)
    @step
    async def evaluate_and_judge(self, ev: RetrievalEvent) -> EvaluationEvent | StopEvent:
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(query=ev.query, context=ev.context[:2000])
        evaluation = str(Settings.llm.complete(judge_prompt)).strip().upper()
        
        # If the judge fails or context is empty, try rewriting query up to 2 times
        if ("FAIL" in evaluation or not ev.context.strip()) and ev.attempt < 2:
            rewriter_prompt = REWRITER_PROMPT_TEMPLATE.format(query=ev.query)
            new_query = str(Settings.llm.complete(rewriter_prompt)).strip()
            new_query = new_query.replace("```", "").replace('"', "").replace("Rewritten:", "").strip()
            return EvaluationEvent(query=new_query, context=ev.context, sources=ev.sources, evaluation="FAIL", attempt=ev.attempt)
        
        return StopEvent(result={"context": ev.context, "query": ev.query, "sources": ev.sources, "attempts": ev.attempt})

# 5. Run the Workflow with CLI Argument Support
async def main():
    parser = argparse.ArgumentParser(description="Run Agentic RAG workflow with a dynamic test query.")
    parser.add_argument("--query", type=str, default="How does Apache Iceberg handle atomic transactions and snapshot isolation?", help="The technical query to run.")
    args = parser.parse_args()

    workflow = AgenticAnalystWorkflow(timeout=120, verbose=False)
    
    print(f"\n🚀 Starting Agentic Workflow for Query: '{args.query}'\n")
    result = await workflow.run(query=args.query)
    
    print("\n--- ✨ Final Synthesized Answer ---\n")
    generation_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(query=result['query'], context=result['context'][:3000])
    final_answer = Settings.llm.complete(generation_prompt)
    print(final_answer)
    
    print("\n--- 🔗 Source Citations & Highlights ---")
    for source in set(result['sources']):
        print(f"• Document: {source}")
    
    print(f"\nSnippet Highlight Preview:\n\"{result['context'][:300]}...\"")
    print(f"\n📊 [Workflow Stats]: Completed successfully in {result['attempts']} attempt(s).\n")

if __name__ == "__main__":
    asyncio.run(main())