import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from engine.retriever2 import RAGRetriever

load_dotenv()

def start_case_chat(case_id="intel_docs"):
    # Initialize retriever for this specific case
    retriever = RAGRetriever(collection_name=case_id)
    
    # Simple in-memory history for this session
    chat_history = [] 

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    print(f"📁 Switched to context for [Case: {case_id}]")
    
    while True:
        query = input("\n👤 You: ")
        if query.lower() in ['exit', 'quit']: break

        # 1. Retrieve context ONLY from this case
        context, sources = retriever.get_relevant_context(query)

        # 2. Build the history-aware system prompt
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in chat_history[-5:]])
        
        system_prompt = f"""
        You are an assistant for Case Investigator. Answer using the context provided.
        
        PREVIOUS CONVERSATION:
        {history_str}
        
        CURRENT CONTEXT:
        {context}
        """

        # 3. Call LLM
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        answer = response.choices[0].message.content
        print(f"\n🤖 Bot: {answer}")
        print(f"📚 Sources: {', '.join(sources)}")

        # 4. Save to history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    cid = input("Enter Case ID to join: ")
    start_case_chat(cid)