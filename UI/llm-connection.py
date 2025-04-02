from langchain_openai import ChatOpenAI
import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
API_BASE = "https://api.deepseek.com/v1"
AWANLLM_API_KEY = "sk-"
MODEL_NAME = "deepseek-chat"

# Set up LangChain OpenAI client for chat completions
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_base=API_BASE,
    openai_api_key=AWANLLM_API_KEY,
    temperature=0.7
)

# Test the chat model
print("Testing chat completion...")
chat_response = llm.invoke("What are vector embeddings?")
print(f"Chat response: {chat_response}")

# Set up LangChain OpenAIEmbeddings for embeddings
# Note: We're using the same model, but the API might need to handle this differently
print("\nTesting embeddings...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Test embedding generation
    test_text = "This is a test document about retrieval augmented generation."
    embedding_vector = embeddings.embed_query(test_text)

    print(f"Success! Embedding vector length: {len(embedding_vector)}")
    print(f"Sample values: {embedding_vector[:5]}...")

except Exception as e:
    print(f"Error with embeddings: {str(e)}")
    print("\nPossible solutions:")
    print("1. Check if your API provider supports embeddings with this model")
    print("2. Look for a dedicated embeddings model in your provider's documentation")
    print("3. Try the /embeddings endpoint directly with a simple request:")
    print("""
    import requests

    response = requests.post(
        "https://api.awanllm.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {AWANLLM_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "Meta-Llama-3.1-70B-Instruct",
            "input": "This is a test document."
        }
    )
    print(response.json())
    """)
    print("\n4. As a fallback, you can use local embeddings with sentence-transformers:")
    print("""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    local_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    embedding_vector = local_embeddings.embed_query("This is a test document.")
    """)