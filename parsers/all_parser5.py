import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from docling.document_converter import (
    DocumentConverter, 
    PdfFormatOption, 
    WordFormatOption, 
    PowerpointFormatOption,
    ImageFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableStructureOptions,
    TableFormerMode
)
from docling_core.types.doc.document import ImageRefMode


load_dotenv()


class SmartDocumentParser:

    def __init__(self, output_dir="data/output", max_workers=None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers or os.cpu_count()

        # ===============================
        # Azure Vision Setup
        # ===============================
        az_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_url = "https://newdocintel.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview"


        pic_options = PictureDescriptionApiOptions(
            url=azure_url,
            headers={"api-key": az_api_key},
            params={"model": "gpt-4.1"},
            concurrency=4,
            prompt=(
                "Describe this image clearly in one paragraph. "
                "If it contains charts or tables, summarize key insights."
            ),
            timeout=30,
        )

        # ===============================
        # Accurate table configuration
        # ===============================
        table_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True
        )

        pdf_pipeline = ThreadedPdfPipelineOptions(
            num_threads=os.cpu_count(),

            # Enable vision globally
            do_picture_description=True,
            picture_description_options=pic_options,
            enable_remote_services=True,
            generate_picture_images=True,
            # force_full_page_ocr=False,

            # OCR enabled for image inputs
            do_ocr=True,

            # Accurate tables
            do_table_structure=True,
            table_structure_options=table_options,

            images_scale=1.0,
            layout_batch_size=8,
            table_batch_size=8,
        )

        # IMPORTANT:
        # In Docling 2.x, pipeline options passed globally apply
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.XLSX,
                InputFormat.HTML,
                InputFormat.IMAGE,
                InputFormat.CSV,
                InputFormat.MD
            ],
            format_options={
        # PDFs use the specialized PDF pipeline
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline),
        
        # Word and PPTX use the standard options but can still use the same pipeline
        InputFormat.DOCX: WordFormatOption(pipeline_options=pdf_pipeline),
        InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pdf_pipeline),
        
        # Images (JPG, PNG) also need this to trigger the Azure Vision description
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_pipeline),
    }
        )

    # ==========================================================
    # PUBLIC METHODS
    # ==========================================================

    def process(self, file_path):
        file_path = Path(file_path)

        try:
            print(f"🔎 Parsing: {file_path.name}")

            if file_path.suffix.lower() == ".txt":
            # 1. Read the raw text
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                
                # 2. Feed it to Docling as a "String" (telling it it's MD)
                # This returns a standard Docling 'ConversionResult'
                result = self.converter.convert_string(raw_text, format=InputFormat.MD)
                document = result.document
            else:
                result = self.converter.convert(str(file_path))
                document = result.document

            return self._save_outputs(document, file_path)

        except Exception:
            print(f"❌ ERROR processing {file_path.name}")
            print(traceback.format_exc())
            return None

    def process_batch(self, file_list):
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process, f): f for f in file_list}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"⚠ Batch error: {e}")

        return results

    # ==========================================================
    # SAVE OUTPUTS
    # ==========================================================

    def _save_outputs(self, document, file_path):

        doc_name = file_path.stem.replace(" ", "_")
        base_dir = (self.output_dir / doc_name).resolve()

        md_dir = (base_dir / "markdown").resolve()
        img_dir = (base_dir / "images").resolve()
        json_dir = (base_dir / "structured").resolve()

        md_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        md_file = md_dir / f"{doc_name}_enriched.md"

        # Save markdown
        document.save_as_markdown(
            filename=md_file,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=img_dir,
            include_annotations=True
        )

        # Save structured JSON
        json_file = json_dir / f"{doc_name}_structured.json"

        structured_payload = {
            "metadata": {
                "source_file": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "parsed_timestamp": datetime.utcnow().isoformat()
            },
            "document": document.export_to_dict()
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(structured_payload, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {doc_name}")

        return {
            "markdown": str(md_file),
            "json": str(json_file),
            "images": str(img_dir)
        }
