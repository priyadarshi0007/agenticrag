import os
import json
import hashlib
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 1. Setup Local Embedding Model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None 

# 2. Setup Persistent Local Vector Store
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("whitepapers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

REGISTRY_PATH = "./ingest_registry.json"

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

# 3. Configure Docling Options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False  
pipeline_options.do_table_structure = True 

converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

registry = load_registry()
docs_for_index = []
data_folder = ".data"

print("🧠 Checking Document State & Hashing Registry...")
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_folder, filename)
        current_hash = compute_file_hash(filepath)
        
        # Check if file is new or modified
        if filename in registry and registry[filename] == current_hash:
            print(f"⏩ Unchanged: {filename} (Skipping re-ingestion)")
            continue
        
        print(f"📄 Processing modified/new file: {filename}...")
        result = converter.convert(filepath)
        markdown_text = result.document.export_to_markdown()
        
        doc = Document(
            text=markdown_text,
            metadata={"source_file": filename}
        )
        docs_for_index.append(doc)
        registry[filename] = current_hash
        print(f"✅ Successfully parsed {filename}!")

if docs_for_index:
    print(f"📦 Embedding and Indexing {len(docs_for_index)} new/updated documents into ChromaDB...")
    index = VectorStoreIndex.from_documents(
        docs_for_index, 
        storage_context=storage_context,
        show_progress=True
    )
    save_registry(registry)
    print("🚀 Incremental Ingestion complete! Registry updated.")
else:
    print("✨ No document changes detected. Vector store is up to date.")