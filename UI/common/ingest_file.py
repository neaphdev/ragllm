import os, tempfile
import streamlit as st
import pandas as pd
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from common.chroma_db_settings import Chroma
from common.constants import CHROMA_SETTINGS
from chromadb.utils import embedding_functions  # Add this import

custom_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_base="https://api.awanllm.com/v1",  # Base URL for the API
    api_key="",  # Your API key
    model_name="Meta-Llama-3.1-70B-Instruct"  # The model you specified
)

# Option 2: Using a free alternative - Sentence Transformers


# Choose which embedding function to use
embedding_function = custom_ef  # Change to custom_ef if your API supports embeddings

# Load environment variables
source_directory = os.environ.get('SOURCE_DIRECTORY', 'documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
collection_name = 'vectordb'
collection = CHROMA_SETTINGS.get_or_create_collection(name='vectordb',embedding_function=embedding_function)
chunk_size = 500
chunk_overlap = 50

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}


def load_single_document(uploaded_file) -> List[Document]:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in LOADER_MAPPING:
        # Generar un nombre único para el archivo temporal
        tmp_filename = f"{uploaded_file.name}"
        tmp_path = os.path.join(tempfile.gettempdir(), tmp_filename)

        # Guardar temporalmente el archivo cargado con el nombre único
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())

        try:
            # Crear una instancia del cargador correspondiente
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(tmp_path, **loader_args)
            return loader.load()
        finally:
            # Eliminar el archivo temporal después de usarlo
            os.unlink(tmp_path)

    raise ValueError(f"Unsupported file extension '{ext}'")



def get_unique_sources_df(chroma_settings:chromadb.ClientAPI):
    try:
        # Get data from the collection
        result = chroma_settings.get_collection('vectordb').get(include=['embeddings', 'documents', 'metadatas'])

        # Check if there are any metadatas
        if not result['metadatas'] or len(result['metadatas']) == 0:
            # Return an empty DataFrame if no data exists
            return pd.DataFrame(columns=['source'])

        # Create DataFrame with the data
        df = pd.DataFrame(result)

        # Extract sources from metadatas
        sources = df['metadatas'].apply(lambda x: x.get('source', None)).dropna().unique()

        # Get only the filenames from paths
        file_names = [source.split('/')[-1] for source in sources]

        # Create DataFrame with unique source filenames
        unique_sources_df = pd.DataFrame(file_names, columns=['source'])

        return unique_sources_df
    except Exception as e:
        # Handle any exceptions and return empty DataFrame
        print(f"Error retrieving sources from vector database: {str(e)}")
        return pd.DataFrame(columns=['source'])


# Modificar process_file para recibir el archivo cargado y el nombre
def process_file(uploaded_file, file_name):
    files_in_vectordb = get_unique_sources_df(CHROMA_SETTINGS)['source'].tolist()
    if file_name in files_in_vectordb:
        return None
    else:
        # Convertir los bytes a documentos de texto
        print(f"Loading file: {file_name=}")
        documents = load_single_document(uploaded_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts


def does_vectorstore_exist(settings) -> bool:
    """
    Checks if vectorstore exists
    """
    collection = settings.get_or_create_collection(collection_name)
    return collection


def ingest_file(uploaded_file, file_name):
    if does_vectorstore_exist(CHROMA_SETTINGS):
        db = Chroma(embedding_function=embeddings, client=CHROMA_SETTINGS)
        texts = process_file(uploaded_file, file_name)
        if texts == None:
            st.warning('Este archivo ya fue agregado anteriormente.')
        else:
            st.spinner(f"Creando embeddings.")
            db.add_documents(texts)
            st.success(f"Se agregó el archivo con éxito.")
    else:
        # Create and store locally vectorstore
        st.success("Creating new vectorstore")
        texts = process_file(uploaded_file, file_name)
        st.spinner(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, client=CHROMA_SETTINGS)
        st.success(f"Se agregó el archivo con éxito.")


def delete_file_from_vectordb(filename:str):
    new_filename = '/tmp/' + filename
    try:
        collection.delete(where={"source": new_filename})
        print(f'Se eliminó el archivo: {filename} con éxito')
    except:
        print(f'Ocurrió un error al eliminar el archivo {filename}')