import os
from pathlib import Path
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling_core.types.doc.document import ImageRefMode


load_dotenv()


class SmartDocumentParser:
    def __init__(self, output_dir="data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Azure Config (only if you need image descriptions) ----
        az_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        azure_url = "https://newdocintel.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview"


        pic_options = PictureDescriptionApiOptions(
            url=azure_url,
            headers={"api-key": az_api_key},
            prompt="Describe this image in a single paragraph.",
            params={"model": "gpt-4.1"},
            timeout=60
        )

        # ---- PDF specific pipeline ----
        pdf_pipeline = PdfPipelineOptions(
            do_picture_description=True,
            picture_description_options=pic_options,
            enable_remote_services=True,
            generate_picture_images=True,
            do_ocr=True,
            do_table_structure=True
        )

        # 🚀 IMPORTANT:
        # Only configure PDF.
        # Let Docling auto-handle other formats.
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline
                )
            }
        )

    def process(self, file_path):
        file_path = Path(file_path)
        print(f"--- Converting: {file_path.name} ---")

        result = self.converter.convert(file_path)

        # Create document-specific folder
        doc_stem = file_path.stem.replace(" ", "_")
        doc_root = self.output_dir / doc_stem
        images_dir = doc_root / "images"

        doc_root.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        md_file = doc_root / f"{doc_stem}_enriched.md"

        result.document.save_as_markdown(
            filename=md_file.resolve(),
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=images_dir,  # keep relative behavior
            include_annotations=True
        )

        print(f"Saved to: {doc_root}")
        return md_file
