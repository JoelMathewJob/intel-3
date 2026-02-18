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

    def get_relevant_context(self, query,  mode="ssr",k=5, source_file=None, threshold=1.0):
        """
        Args:
            mode: "mmr" for summaries/diversity, "similarity" for lists/extraction.
            source_file: Filename to filter by (e.g. "Career_Cyber_Security.pdf").
            threshold: 0.0 to 2.0. (Lower is stricter). 1.0 is a safe bet for MiniLM.
        """
        # 1. Setup metadata filter
        search_filter = {"source_file": source_file} if source_file else None

        # 2. Execute Search with Scores
        if mode == "mmr":
            # Langchain's MMR doesn't return scores directly, 
            # so we fetch docs and then check scores via a helper or threshold search
            results = self.vector_db.max_marginal_relevance_search(
                query, k=k, fetch_k=20, filter=search_filter, lambda_mult=0.5
            )
            # For MMR, we usually trust the k since it's for general chat
            final_docs = results
        else:
            # Similarity Search with Score (Best for UPI/Email lists)
            results_with_scores = self.vector_db.similarity_search_with_score(
                query, k=k, filter=search_filter
            )
            # Filter by Distance (Handles the "Hi" problem)
            final_docs = [doc for doc, score in results_with_scores if score <= threshold]

        if not final_docs:
            return "I couldn't find any relevant information in the documents.", []

        # 3. Build Context String
        context = ""
        for doc in final_docs:
            src = doc.metadata.get("source_file", "Unknown")
            context += f"--- Document: {src} ---\n{doc.page_content}\n\n"
        print("retrived chunks:",context)
        sources = list(set([doc.metadata.get("source_file") for doc in final_docs]))
        return context, sources
    
    