import os
import numpy as np
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from engine.vector_db import VectorEngine

class RAGRetriever:
    def __init__(self, collection_name="default"):
        self.engine = VectorEngine()
        self.collection_name = collection_name
        
        # Initialize Vector DB
        self.vector_db = Chroma(
            persist_directory=self.engine.persist_directory,
            embedding_function=self.engine.embeddings,
            collection_name=self.collection_name
        )

        # Initialize LLM for Multi-Query expansion
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _get_retrieval_mode(self, query):
        """Automatically decides mode based on keyword triggers."""
        query_lower = query.lower()
        extraction_keywords = ["list", "all", "every", "contact", "email", "upi", "phone", "details"]
        
        if any(word in query_lower for word in extraction_keywords):
            return "similarity", 15  # Exhaustive search
        return "mmr", 5             # Diverse search for summaries

    def get_relevant_context(self, query, source_file=None):
        # 1. Decide search mode and k-value
        mode, k = self._get_retrieval_mode(query)
        
        # 2. Setup metadata filter if user specified a file
        search_filter = {"source_file": source_file} if source_file else None

        # 3. Multi-Query Expansion (Solves 'UPI Details' vs 'UPIdetails')
        # This generates 3 variations of the query and runs them in parallel
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_db.as_retriever(
                search_type=mode, 
                search_kwargs={"k": k, "filter": search_filter}
            ),
            llm=self.llm
        )

        # 4. Execute Search
        # similarity_search_with_score allows us to apply the threshold for "Hi"
        if mode == "similarity":
            # For extraction, we use similarity with a strict threshold (e.g., 1.1)
            results_with_scores = self.vector_db.similarity_search_with_score(
                query, k=k, filter=search_filter
            )
            # Thresholding logic: distance <= 1.1 is relevant for MiniLM models
            results = [doc for doc, score in results_with_scores if score <= 1.1]
        else:
            # MMR mode for general chat
            results = mq_retriever.invoke(query)

        if not results:
            print("--- No relevant chunks found above threshold ---")
            return "No relevant information found.", []

        # 5. Build Context and Extract Sources
        context = ""
        for doc in results:
            src = doc.metadata.get("source_file", "Unknown")
            context += f"--- Source: {src} ---\n{doc.page_content}\n\n"
            
        sources = list(set([doc.metadata.get("source_file", "Unknown") for doc in results]))
        
        print(f"Retrieved {len(results)} chunks from {len(sources)} sources.")
        return context, sources