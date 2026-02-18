# no image tags [image]


import os
import re
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class RAGChunker:
    def __init__(self, chunk_size=1500, chunk_overlap=200): # Increased size slightly
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    def _clean_text(self, text):
        """Removes Markdown image tags and local file paths."""
        # 1. Remove ![Image](path/to/file.png)
        text = re.sub(r'!\[Image\]\(.*?\)', '', text)
        
        # 2. Remove any remaining stray image placeholders like [image_001]
        text = re.sub(r'\[image_\d+\]', '', text)
        
        # 3. Remove local Windows/Linux file paths that might be left behind
        # This matches common paths like C:\PA\... or /data/output/...
        text = re.sub(r'[A-Z]:\\(?:[\w\s.-]+\\)*[\w\s.-]+\.(?:png|jpg|jpeg|pdf)', '', text)
        
        # 4. Clean up excessive whitespace/newlines created by removals
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


    def create_chunks(self, md_text, filename):

        clean_text = self._clean_text(md_text)

        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
        is_image = filename.lower().endswith(image_extensions)
        # 1. Markdown Header Split (Keep context grouped)
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False # Keep headers in text so LLM sees them
        )
        header_splits = markdown_splitter.split_text(clean_text)

        if is_image or len(md_text)<2000:
            final_chunks = header_splits
        else:
        # 2. Refined Recursive Splitter
        # Adding "|" to separators ensures we try to avoid splitting inside Markdown tables
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            # Priority: Double newline > Single newline > Table Pipe > Space
            separators=["\n\n", "\n", "|", " ", ""] 
        )
        
            final_chunks = text_splitter.split_documents(header_splits)
        # 3. Inject Metadata (CRITICAL for your filtering)
        for chunk in final_chunks:
            chunk.metadata["source_file"] = filename
        return final_chunks