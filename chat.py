import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from engine.retriever import RAGRetriever

load_dotenv()

def chat_with_docs():
    # 1. Connect to the existing ChromaDB (doesn't re-index anything)
    retriever = RAGRetriever()
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    print("🤖 RAG Bot initialized. I'm reading from your persistent database.")
    print("(Type 'exit' to quit)")
    
    while True:
        query = input("\n👤 You: ")
        if query.lower() in ['exit', 'quit']: break

        # STEP 1: Retrieve chunks + metadata
        # We modify the retriever to return the objects, not just text
        results = retriever.vector_db.similarity_search(query, k=4)
        
        context = "\n\n".join([doc.page_content for doc in results])
        # Collect unique sources
        sources = list(set([doc.metadata.get("source_file", "Unknown") for doc in results]))

        # STEP 2: The Grounded Prompt
        system_prompt = f"""
        You are a professional document assistant. Answer the user's question ONLY using the provided context.
        If the answer is not in the context, politely state that you don't have that information.
        
        CONTEXT:
        {context}
        """

        # STEP 3: Generate Response
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        # STEP 4: Output with Citations
        print(f"\n🤖 Bot: {response.choices[0].message.content}")
        print(f"\n📚 Sources used: {', '.join(sources)}")

if __name__ == "__main__":
    chat_with_docs()