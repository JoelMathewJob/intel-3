import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import shutil

# Lazy load your existing logic
from parsers.all_parser8 import SmartDocumentParser
from engine.chunkers.chunker4 import RAGChunker
from engine.vector_db import VectorEngine
from openai import AzureOpenAI
from dotenv import load_dotenv
from engine.retrievers.retriever2 import RAGRetriever

from langchain_openai import AzureChatOpenAI # Use the LangChain wrapper

# Initialize the LangChain LLM wrapper
llm = AzureChatOpenAI(
    # Use the EXACT name of the deployment from your Azure portal
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), 
    # Must be the base model name (e.g., "gpt-4" or "gpt-35-turbo")
    model_name="gpt-4.1", 
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)


load_dotenv()

app = FastAPI(title="Case Intelligence API")


# Allow React to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (Loaded ONCE on startup)
parser = SmartDocumentParser(output_dir="data/output")
chunker = RAGChunker(chunk_size=1500, chunk_overlap=200)

class ChatRequest(BaseModel):
    message: str
    case_id: str
    history: Optional[List[dict]] = []

class DocSummarize(BaseModel):
    case_id: str
    filename: str

@app.post("/ingest")
async def ingest_files(case_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Uploads and processes files using your ingestion pipeline logic."""
    try:
        # 1. Setup directories
        input_dir = Path(f"data/input/{case_id}")
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Initialize Vector DB for this specific case
        vector_db = VectorEngine(collection_name=case_id)
        
        results = []
        for file in files:
            # Save the file locally
            file_path = input_dir / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # --- START PIPELINE (Same as your process_single_file) ---
            # Step A: Parsing
            parsed_results = parser.process(file_path)
            if not parsed_results or "markdown" not in parsed_results:
                results.append(f"FAILED: {file.filename} (Parsing issue)")
                continue

            # Step B: Read Parsed Content
            md_path = Path(parsed_results["markdown"])
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Step C: Chunking
            chunks = chunker.create_chunks(content, file.filename)
            # for chunk in chunks:
            #     chunk.metadata["source_file"] = file.filename
            
            # Step D: Storing
            success = vector_db.store_documents(chunks)
            if success:
                results.append(f"SUCCESS: {file.filename}")
            else:
                results.append(f"PARTIAL SUCCESS: {file.filename} (Indexing failed)")

        print("results:",results)   
        return {"status": "success", "details": results}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/chat")
async def chat(req: ChatRequest):
    """Handles RAG retrieval and GPT-4 response."""
    # Initialize engine for this specific case
    retriever = RAGRetriever(collection_name=req.case_id)
    
    # 1. Retrieve (Use your smarter logic here)
    context, sources = retriever.get_relevant_context(req.message)
    # context = "\n\n".join([f"SOURCE {d.metadata['source_file']}: {d.page_content}" for d in results])
    
    # 2. Azure OpenAI Call
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    system_prompt = f"You are a Case Investigator , Use the context to answer. If unsure, say you don't know.\n\nCONTEXT:\n{context}"
    
    messages = [{"role": "system", "content": system_prompt}]
    # messages.extend(req.history[-5:]) # Add last 5 messages for history
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="gpt-4.1", # Use your actual deployment name
        messages=messages
    )

    answer = response.choices[0].message.content
    
    # sources = list(set([d.metadata['source_file'] for d in results]))

    return {"answer": answer, "sources": sources}


@app.post("/doc_summarize")
async def doc_summarize(req:DocSummarize):
    map_prompt_template = """
    Extract the most critical key points from the following text:
    "{text}"
    Key Points:
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # The combine prompt generates the final output
    # This is where we strictly enforce brevity and ban filler words
    combine_prompt_template = """
    Write a very brief, direct summary of the following text. 
    CRITICAL INSTRUCTIONS: Do NOT use introductory phrases like "Here is a concise summary", "In summary", "This text discusses", or any similar filler. Just give the raw facts directly. Keep it to 3-4 short sentences max.

    "{text}"

    Brief Summary:
    """
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
    try:
        # 1. Initialize retriever for the specific case
        retriever = RAGRetriever(collection_name=req.case_id)
        
        # 2. Get all chunks for the file
        docs = retriever.get_chunks(req.filename)
        
        if not docs:
            raise HTTPException(status_code=404, detail=f"No chunks found for file: {req.filename}")

        # 3. Load the summarization chain
        # "map_reduce" is best for large docs as it summarizes chunks separately first
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)
        
        # 4. Run summarization
        summary = await chain.ainvoke(docs)
        
        return {
            "case_id": req.case_id,
            "filename": req.filename,
            "answer": summary["output_text"]
        }

    except Exception as e:
        print(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)