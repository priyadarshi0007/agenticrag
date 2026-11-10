from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 1. Load the exact same local embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

# 2. Connect to your existing Chroma database
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("whitepapers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. Load the index from storage
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# 4. Run a test query against your whitepapers
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What are the core benefits of Lakebase ?")

print("\n--- 🔍 Vector Search Verification Result ---\n")
print(response)