from langchain_chroma import Chroma
from engine.vector_db import VectorEngine

class RAGRetriever:
    def __init__(self, collection_name="default"):
        self.engine = VectorEngine()
        self.collection_name = collection_name
        
        # Connect to the SPECIFIC collection for this case
        self.vector_db = Chroma(
            persist_directory=self.engine.persist_directory,
            embedding_function=self.engine.embeddings,
            collection_name=self.collection_name
        )

    def get_relevant_context(self, query, k=4):
        results = self.vector_db.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        sources = list(set([doc.metadata.get("source_file", "Unknown") for doc in results]))
        return context, sources