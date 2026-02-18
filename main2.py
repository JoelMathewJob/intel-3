import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# from parsers.all_parser8 import SmartDocumentParser
# from engine.chunker2 import RAGChunker
# from engine.vector_db import VectorEngine

load_dotenv()

def process_single_file(file_path, parser, chunker, vector_db):
    """Handles the full pipeline for one file."""
    try:
        # STEP A: Extraction
        parsed_results = parser.process(file_path)
        
        if not parsed_results or "markdown" not in parsed_results:
            return f"FAILED: {file_path.name} (Parsing issue)"

        md_path = Path(parsed_results["markdown"])
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # STEP B & C: Chunking & Metadata
        chunks = chunker.create_chunks(content, file_path.name)
        # for chunk in chunks:
        #     chunk.metadata["source_file"] = file_path.name
        
        # STEP D: Indexing (Wrapped in try-except in vector_db)
        success = vector_db.store_documents(chunks)
        if success:
            return f"SUCCESS: {file_path.name}"
        else:
            return f"PARTIAL SUCCESS: {file_path.name} (Parsed but Indexing failed)"

    except Exception as e:
        return f"ERROR processing {file_path.name}: {str(e)}"

def run_ingestion_pipeline():
    # Initialize components
    case_id = input("enter collection name: ")

    
    from parsers.all_parser8 import SmartDocumentParser
    from engine.chunkers.chunker4 import RAGChunker
    from engine.vector_db import VectorEngine
    
    parser = SmartDocumentParser(output_dir="data/output")
    chunker = RAGChunker(chunk_size=800, chunk_overlap=80)
    # vector_db = VectorEngine(collection_name=collection_name)
    vector_db = VectorEngine(collection_name=case_id)

    input_folder = Path("data/input")
    files_to_process = [f for f in input_folder.glob("*")]

    print(f"🚀 Starting parallel ingestion for {len(files_to_process)} files...\n")

    # Use ThreadPoolExecutor for parallel parsing
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_file, f, parser, chunker, vector_db): f for f in files_to_process}
        
        for future in as_completed(futures):
            result = future.result()
            print(result)

    print("\n✅ Ingestion cycle complete.")

if __name__ == "__main__":
    run_ingestion_pipeline()