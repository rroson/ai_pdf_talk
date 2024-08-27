import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

load_dotenv()

# Mean Pooling - Considera a máscara de atenção para a média correta
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # O primeiro elemento de model_output contém todos os embeddings de tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_docs(directory):
    """Carrega todos os documentos de um diretório"""
    loader = DirectoryLoader(path=directory, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def split_docs(documents):
    """Divide um documento em pequenos pedaços"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

def embed_sentences(sentences, tokenizer, model):
    """Gera embeddings para uma lista de sentenças"""
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def insert_data():
    """Cria embeddings para os pedaços de documentos e os insere no Milvus DB"""
    try:
        # Carrega documentos e os divide em pedaços
        documents = load_docs("./pdf_documents")
        docs = split_docs(documents)

        # Inicializa o modelo e tokenizer do Hugging Face
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # Processa cada documento e gera embeddings
        embeddings = []
        for doc in docs:
            sentence_embedding = embed_sentences([doc.page_content], tokenizer, model)
            embeddings.append(sentence_embedding)

        # Converte os embeddings para o formato adequado
        embeddings = torch.cat(embeddings).numpy()

        # Insere os embeddings no Milvus
        Milvus.from_documents(
            docs,
            embeddings,
            collection_name=os.getenv("MILVUS_DB_COLLECTION"),
            connection_args={
                "user": os.getenv("MILVUS_DB_USERNAME"),
                "password": os.getenv("MILVUS_DB_PASSWORD"),
                "host": os.getenv("MILVUS_DB_HOST"),
                "port": os.getenv("MILVUS_DB_PORT"),
                "db_name": os.getenv("MILVUS_DB_NAME")
            }
        )
        print("Arquivo inserido no banco de dados vetorial com sucesso")
    except Exception as exception_message:
        print(str(exception_message))

if __name__ == "__main__":
    insert_data()
