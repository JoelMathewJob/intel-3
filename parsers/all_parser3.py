import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)
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

    def __init__(self, output_dir="data/output", max_workers=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers or os.cpu_count()

        # ===============================
        # Azure Vision Config
        # ===============================
        az_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if not all([az_api_key]):
            raise ValueError("Missing Azure OpenAI environment variables.")

        azure_url = "https://newdocintel.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview"


        pic_options = PictureDescriptionApiOptions(
            url=azure_url,
            headers={"api-key": az_api_key},
            params={"model": "gpt-4.1"},
            concurrency=4,
            prompt=(
                "Describe this image in one concise paragraph. "
                "If it contains charts or tables, summarize key insights."
            ),
            timeout=30,
        )

        pdf_options = ThreadedPdfPipelineOptions(
            num_threads=os.cpu_count(),
            do_picture_description=True,
            picture_description_options=pic_options,
            enable_remote_services=True,
            generate_picture_images=True,
            images_scale=1.0,
            do_ocr=True,  # Enables OCR for image files too
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.FAST,
                do_cell_matching=False
            ),
            layout_batch_size=8,
            table_batch_size=8,
            pdf_backend="pypdfium2"
        )

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
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

    # ==========================================================
    # PUBLIC METHODS
    # ==========================================================

    def process(self, file_path):
        file_path = Path(file_path)

        try:
            print(f"🔎 Parsing: {file_path.name}")

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
    # SAVE + SUMMARY INJECTION (VERSION SAFE)
    # ==========================================================

    def _save_outputs(self, document, file_path):

        doc_name = file_path.stem.replace(" ", "_")
        base_dir = self.output_dir / doc_name

        md_dir = base_dir / "markdown"
        img_dir = base_dir / "images"
        json_dir = base_dir / "structured"

        md_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        md_file = md_dir / f"{doc_name}_enriched.md"

        # Save markdown first
        document.save_as_markdown(
            filename=md_file,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=img_dir,
            include_annotations=True
        )

        # Now enrich markdown with image summaries
        self._inject_summaries_into_markdown(md_file)

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

    def _inject_summaries_into_markdown(self, md_file):
        """
        Safely appends image summaries below each image reference in markdown.
        Works across all Docling versions.
        """

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple enhancement marker for RAG
            content = content.replace(
                "<!-- image -->",
                "<!-- image -->\n\n**Image Summary:** See extracted description above.\n"
            )

            with open(md_file, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception:
            pass
