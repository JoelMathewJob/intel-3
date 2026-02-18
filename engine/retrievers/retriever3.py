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

    def get_relevant_context(self, query, k=8):
    # MMR search for better diversity in context
    # fetch_k: number of docs to initially find
    # lambda_mult: 0.5 is balanced, 0 is max diversity, 1 is max similarity
        results = self.vector_db.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=20, 
            lambda_mult=0.5
        )
        print("retirved chunks:",results,"\n\n")
        
        context = "\n\n".join([doc.page_content for doc in results])
        sources = list(set([doc.metadata.get("source_file", "Unknown") for doc in results]))
        return context, sources