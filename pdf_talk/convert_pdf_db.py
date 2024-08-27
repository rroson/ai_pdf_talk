"""Convert a document into smaller chunks of text, vectorize it 
and store it into the Milvus DB"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

def load_docs(directory):
    """Loads all documents from a directory"""
    loader = DirectoryLoader(path=directory, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def split_docs(documents):
    """Splits a document into small chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
    return text_splitter.split_documents(documents)


def insert_data():
    """Creates embeddings for document chunks and inserts them into Milvus DB"""
    try:
        documents = load_docs("./pdf_documents")
        docs = split_docs(documents)
        embeddings = OpenAIEmbeddings()
        Milvus.from_documents(docs, embeddings,
                              collection_name=os.getenv("MILVUS_DB_COLLECTION"),
                              connection_args={
                                  "user": os.getenv("MILVUS_DB_USERNAME"),
                                  "password": os.getenv("MILVUS_DB_PASSWORD"),
                                  "host": os.getenv("MILVUS_DB_HOST"),
                                  "port": os.getenv("MILVUS_DB_PORT"),
                                  "db_name": os.getenv("MILVUS_DB_NAME")})
        print("File inserted into vector database successfully")
    except Exception as exception_message:
        print(str(exception_message))


if __name__ == "__main__":
    insert_data()
