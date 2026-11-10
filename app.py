import asyncio
import importlib.util
import sys
import streamlit as st
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from prompts import SYNTHESIS_PROMPT_TEMPLATE

# Dynamically load your workflow script
spec = importlib.util.spec_from_file_location("workflow_module", "3_agentic_workflow.py")
workflow_module = importlib.util.module_from_spec(spec)
sys.modules["workflow_module"] = workflow_module
spec.loader.exec_module(workflow_module)

AgenticAnalystWorkflow = workflow_module.AgenticAnalystWorkflow

# Configure Local M1 Pro Stack Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0)

# Streamlit Page Config
st.set_page_config(
    page_title="Local Agentic RAG Architecture", 
    page_icon="💻", 
    layout="centered"
)

# Header Render
st.title("Local Agentic RAG Architecture")
st.markdown("Air-gapped technical analysis powered by Docling, ChromaDB, LlamaIndex Workflows, and Gemma-2b.")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

async def run_workflow_async(query: str):
    workflow = AgenticAnalystWorkflow(timeout=120, verbose=False)
    result = await workflow.run(query=query)
    return result

# Handle new user input
if prompt := st.chat_input("Ask a technical question (e.g., How does Iceberg handle atomic transactions?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # Execute Workflow inside Event Loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_workflow_async(prompt))

        # RENDER INTERACTIVE HORIZONTAL PROCESS FLOW
        st.markdown("### Execution Pipeline")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Phase 1", value="Ingestion")
            with st.expander("Inspect Query"):
                st.code(result['query'], language="text")
                
        with col2:
            st.metric(label="Phase 2", value="Retrieval", delta=f"{len(set(result['sources']))} file(s)")
            with st.expander("Inspect Sources"):
                for src in set(result['sources']):
                    st.markdown(f"- `{src}`")
                    
        with col3:
            st.metric(label="Phase 3", value="Judge", delta=f"{result['attempts']} attempt(s)")
            with st.expander("Inspect Judge"):
                st.write(f"• Loop Iterations: {result['attempts']}")
                st.write("• Status: Grounded Validation Complete")
                
        with col4:
            st.metric(label="Phase 4", value="Synthesis")
            with st.expander("Inspect Context"):
                st.text(result['context'][:800] + "\n...")

        st.divider()

        # Generate Final Answer
        generation_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(query=result['query'], context=result['context'][:3000])
        final_answer = str(Settings.llm.complete(generation_prompt)).strip()

        sources_str = "\n".join([f"- `{s}`" for s in set(result['sources'])])
        
        response_content = f"""
### Synthesized Analysis
{final_answer}

**Source Citations:**
{sources_str}
        """

        st.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})