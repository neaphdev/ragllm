import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
# Create an embedding function
# Option 1: API-based embeddings (using your custom endpoint)
# Note: Make sure your API endpoint supports embedding generation, not just chat completions


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Option 2: Using a free alternative - Sentence Transformers
# If you don't have or want to use API-based embeddings
print("conected")
"""
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # This is a lightweight model
)
"""

# Choose which embedding function to use
embedding_function = embeddings  # Change to custom_ef if your API supports embeddings

# Initialize the ChromaDB client
chroma_client = chromadb.HttpClient(
    host='10.255.255.254',  # Use 'host.docker.internal' if running in a Docker container
    port=8000,
    settings=Settings()
)
# First, get a reference to an existing collection
collection = chroma_client.get_collection(name="test_collection2")

# Then query the collection
results = collection.query(
    query_texts=["can you provide me 1rst document"],
    n_results=5,
    include=[]
)

print(results)