import chromadb
from chromadb.config import Settings

# Define the folder for storing database
#PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')

# Define the Chroma settings
CHROMA_SETTINGS = chromadb.HttpClient(host="10.255.255.254", port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
print(f"{'*'*10} ",CHROMA_SETTINGS.list_collections())