import os
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
import chromadb
from langchain_chroma import Chroma

chromadb.api.client.SharedSystemClient.clear_system_cache()


class VectorEngine:
    def __init__(self, collection_name="intel_docs"):
        # This model is free, runs locally, and is very fast
        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {'device': 'cpu'} # Change to 'cuda' if you have a GPU
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.persist_directory = "./data/chroma_db"
        self.collection_name  = collection_name

    def store_documents(self, chunks):
        try:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            return vector_db
        except Exception as e:
            print(f"❌ Error indexing to Chroma: {e}")
            return None