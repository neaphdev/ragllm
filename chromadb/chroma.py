import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions  # Add this import
# Create an embedding function
# Option 1: API-based embeddings (using your custom endpoint)
# Note: Make sure your API endpoint supports embedding generation, not just chat completions


custom_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_base="https://api.awanllm.com/v1",  # Base URL for the API
    api_key="67e7ee0b-64d1-47a8-9f3e-1d4223ce7afb",  # Your API key
    model_name="Meta-Llama-3.1-70B-Instruct"  # The model you specified
)

# Option 2: Using a free alternative - Sentence Transformers
# If you don't have or want to use API-based embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # This is a lightweight model
)

# Choose which embedding function to use
embedding_function = sentence_transformer_ef  # Change to custom_ef if your API supports embeddings

# Initialize the ChromaDB client
chroma_client = chromadb.HttpClient(
    host='localhost',  # Use 'host.docker.internal' if running in a Docker container
    port=8000,
    settings=Settings()
)

# Create a new collection
collection = chroma_client.create_collection(
    name='test_collection2',
    embedding_function=embedding_function  # Pass the embedding function here
)

# Add a document to the collection
collection.add(
    ids=['1', '2', '3'],
    documents=[
        'This is the first document',
        'This is the second document',
        'This is the third document',
    ]
)

# Search for documents similar to the first document
similar_documents = collection.query(
    query_texts=['This is the first document'],
    n_results=2,
    include=["documents"]
)

print(similar_documents)