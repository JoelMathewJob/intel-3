import os
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class RAGChunker:
    def __init__(self, chunk_size=1500, chunk_overlap=150): # Increased size slightly
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    
    def create_chunks(self, md_text):
        # 1. Markdown Header Split (Keep context grouped)
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False # Keep headers in text so LLM sees them
        )
        header_splits = markdown_splitter.split_text(md_text)

        # 2. Refined Recursive Splitter
        # Adding "|" to separators ensures we try to avoid splitting inside Markdown tables
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", "", "|"] 
        )
        
        final_chunks = text_splitter.split_documents(header_splits)
        return final_chunks