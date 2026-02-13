from langchain_chroma import Chroma
from engine.vector_db import VectorEngine # Reuse your BGE setup

class RAGRetriever:
    def __init__(self, collection_name):
        # Initialize your local BGE embeddings
        self.engine = VectorEngine()
        # Load the existing collection
        self.vector_db = Chroma(
            persist_directory=self.engine.persist_directory,
            embedding_function=self.engine.embeddings,
            collection_name=collection_name # Match your current name
        )

    def get_relevant_context(self, query, k=4):
        # Search the DB for the most similar text chunks
        docs = self.vector_db.similarity_search(query, k=k)
        # Join the text into one big context block
        return "\n\n---\n\n".join([doc.page_content for doc in docs])