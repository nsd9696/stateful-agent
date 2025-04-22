import os
from uuid import uuid4

import chromadb
from dotenv import load_dotenv
from hyperpocket.tool import function_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))


@function_tool
def create_collection(collection_name: str):
    """
    Create a new collection in the Chroma client.
    """
    collection_name = collection_name.lower()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    )
    return vector_store


@function_tool
def add_pdf_documents(collection_name: str, pdf_path: str):
    """
    Add a PDF document to the Chroma client.
    """
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(os.getenv("DEFAULT_DATA_DIR"), pdf_path)
    collection_name = collection_name.lower()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    )
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    return f"Documents {pdf_path} added to collection {collection_name}"


@function_tool
def query_collection(collection_name: str, query: str):
    """
    Query the Chroma client.
    """
    collection_name = collection_name.lower()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    )
    results = vector_store.similarity_search(query)
    return results


@function_tool
def delete_collection(collection_name: str):
    pass


@function_tool
def delete_all_collections():
    pass


@function_tool
def get_collection_names():
    pass


@function_tool
def get_collection_info(collection_name: str):
    pass
