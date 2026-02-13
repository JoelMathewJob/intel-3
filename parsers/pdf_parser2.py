
import os
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions, #
    PictureDescriptionApiOptions,
    TableStructureOptions,
    TableFormerMode
)
from docling_core.types.doc.document import ImageRefMode
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

load_dotenv()

class SmartPDFParser:
    def __init__(self, output_dir="data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. API Configuration (Using your hardcoded details)
        az_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        # Hardcoded URL as requested to avoid construction errors
        azure_url = "https://newdocintel.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview"

        # 2. Configure Docling's Picture Description API
        # Note: 'params' must include the deployment name as the 'model'
        pic_options = PictureDescriptionApiOptions(
            url=azure_url,
            headers={"api-key": az_api_key},
            prompt="Describe this image in a single paragraph. Focus on charts, tables, and data values.",
            params={"model": "gpt-4.1"}, 
            timeout=60,
            concurrency=8 # Process 4 images at once
        )

        # 3. Setup Pipeline Options
        pipeline_options = ThreadedPdfPipelineOptions( # Use threaded version
            do_picture_description=True,
            picture_description_options=pic_options,
            enable_remote_services=True,
            generate_picture_images=True,
            images_scale=2.0,  # Keep scale at 1.0 for speed
            do_ocr=False,      # Turn off OCR if documents are digital
            
            # Use faster table mode
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE, # Accurate is better for large text
                do_cell_matching=True # Forces text to stay inside cell borders
            ),
            
            # Parallel batch sizes
            layout_batch_size=4,
            table_batch_size=4,
            
            # Switch to faster backend
            pdf_backend="pypdfium2"
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    # def process(self, pdf_path):
    #     print(f"--- Converting: {Path(pdf_path).name} ---")
        
    #     # 1. Run the conversion
    #     result = self.converter.convert(pdf_path)
        
    #     # 2. Setup your specific output directories
    #     # Using 'data/output/md' for the markdown files
    #     md_output_dir = self.output_dir / "md"
    #     md_output_dir.mkdir(parents=True, exist_ok=True)

    #     # 3. Define the final markdown filename (matches notebook logic)
    #     doc_stem = Path(pdf_path).stem.replace(" ", "_")
    #     md_filename = md_output_dir / f"{doc_stem}_enriched.md"

    #     # 4. Save using REFERENCED mode (from cell 10 of your notebook)
    #     # This correctly saves images into a subfolder next to the .md file
    #     result.document.save_as_markdown(
    #         md_filename, 
    #         image_mode=ImageRefMode.REFERENCED, 
    #         include_annotations=True
    #     )
        
    #     print(f"--- Success! ---")
    #     print(f"Markdown saved to: {md_filename}")
    #     # Note: Images are automatically saved in {md_filename.stem}_artifacts/
        
    #     return md_filename

    def process(self, pdf_path):
        print(f"--- Converting: {Path(pdf_path).name} ---")
        
        result = self.converter.convert(pdf_path)

        # 1️⃣ Create document root folder
        doc_stem = Path(pdf_path).stem.replace(" ", "_")
        doc_root_dir = (self.output_dir / doc_stem).resolve()
        doc_root_dir.mkdir(parents=True, exist_ok=True)

        # 2️⃣ Images folder inside document root
        images_output_dir = doc_root_dir / "images"
        images_output_dir.mkdir(parents=True, exist_ok=True)

        # 3️⃣ Markdown file directly inside doc root
        md_filename = doc_root_dir / f"{doc_stem}_enriched.md"

        # 4️⃣ Save correctly
        result.document.save_as_markdown(
            filename=str(md_filename.resolve()),
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=images_output_dir,
            include_annotations=True
        )

        # Inside your process function:
        for i, table in enumerate(result.document.tables):
            table_df = table.export_to_dataframe()
            table_df.to_csv(self.output_dir / f"table_{i}.csv")

        print(f"--- Success! ---")
        print(f"Folder: {doc_root_dir}")

        return md_filename
