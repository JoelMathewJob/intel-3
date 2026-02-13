import os
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions, 
    PictureDescriptionApiOptions,
    TableStructureOptions,
    TableFormerMode
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.document import ImageRefMode

load_dotenv()

class SmartDocumentParser:
    def __init__(self, output_dir="data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. API Config for PDFs & Images (Vision required)
        az_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        az_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        az_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_url = f"{az_endpoint}/openai/deployments/{az_deployment}/chat/completions?api-version={az_api_version}"

        pic_options = PictureDescriptionApiOptions(
            url=azure_url,
            headers={"api-key": az_api_key},
            params={"model": az_deployment},
            concurrency=8,
            prompt="Describe this image in a single paragraph. Focus on charts, tables, and data values.",
            timeout=30,

        )

        # 2. PDF Pipeline (The "Heavy" one)
        # pdf_options = ThreadedPdfPipelineOptions(
        #     num_threads=os.cpu_count(),
        #     do_picture_description=True,
        #     picture_description_options=pic_options,
        #     enable_remote_services=True,
        #     generate_picture_images=True,
        #     table_structure_options=TableStructureOptions(
        #         mode=TableFormerMode.ACCURATE, 
        #         do_cell_matching=True
        #     )
        # )


        pdf_options = ThreadedPdfPipelineOptions( # Use threaded version
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


        # 3. Initialize Universal Converter
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, 
                InputFormat.XLSX, InputFormat.HTML, InputFormat.IMAGE,
                InputFormat.CSV, InputFormat.MD
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    backend=PyPdfiumDocumentBackend
                )
                # Word, PPTX, etc. use SimplePipeline by default
            }
        )

    def process(self, file_path):
        print(f"--- Parsing: {Path(file_path).name} ---")
        result = self.converter.convert(file_path)
        
        md_output_dir = self.output_dir / "md"
        images_output_dir = self.output_dir / "images"
        md_output_dir.mkdir(parents=True, exist_ok=True)
        images_output_dir.mkdir(parents=True, exist_ok=True)

        doc_stem = Path(file_path).stem.replace(" ", "_")
        md_filename = md_output_dir / f"{doc_stem}_enriched.md"

        # Export unified Markdown regardless of input type
        result.document.save_as_markdown(
            filename=md_filename, 
            image_mode=ImageRefMode.REFERENCED, 
            artifacts_dir=images_output_dir,
            include_annotations=True
        )
        return md_filename