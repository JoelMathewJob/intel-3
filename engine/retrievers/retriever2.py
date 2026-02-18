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
        results = self.vector_db.similarity_search(query, k=k)
        print("retirved chunks:",results,"\n\n")
        context = "\n\n".join([doc.page_content for doc in results])
        sources = list(set([doc.metadata.get("source_file", "Unknown") for doc in results]))
        return context, sources

    def get_chunks(self, filename: str):
        """
        Retrieves all document chunks associated with a specific filename 
        within the current collection.
        """
        # We query the collection for all items where the metadata 'source_file' matches
        # Chroma's .get() allows us to filter without a query vector
        results = self.vector_db.get(
            where={"source_file": filename}
        )
        
        # results['documents'] is a list of strings
        # We wrap them back into LangChain Document objects for easier chain processing
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content=content, metadata=meta) 
            for content, meta in zip(results['documents'], results['metadatas'])
        ]
        
        return docs