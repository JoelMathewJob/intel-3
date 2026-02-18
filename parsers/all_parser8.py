# video image ocr



import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import requests
import cv2
from dotenv import load_dotenv

from docling.document_converter import (
    DocumentConverter, 
    PdfFormatOption, 
    WordFormatOption, 
    PowerpointFormatOption,
    ImageFormatOption,
    HTMLFormatOption,       # Added
    MarkdownFormatOption,   # Added
    ExcelFormatOption,
    AudioFormatOption     
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableStructureOptions,
    TableFormerMode,
    AsrPipelineOptions,
    RapidOcrOptions
)
from docling_core.types.doc.document import ImageRefMode

from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.datamodel import asr_model_specs



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

        ocr_options = RapidOcrOptions()

        table_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True
        )

        true_pdf_pipeline = ThreadedPdfPipelineOptions( # Use threaded version
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

        doc_pipeline = ThreadedPdfPipelineOptions(

            # Enable vision globally
            do_picture_description=True,
            picture_description_options=pic_options,
            enable_remote_services=True,
            generate_picture_images=True,
            force_full_page_ocr=True,

            # OCR enabled for image inputs
            do_ocr=True,
            ocr_options=RapidOcrOptions(force_full_page_ocr=True),
            # Accurate tables
            do_table_structure=True,
            table_structure_options=table_options,

            images_scale=1.0,
            layout_batch_size=8,
            table_batch_size=8,
            pdf_backend="pypdfium2",


            # audio media
        #     asr_options = AsrPipelineOptions(
        #     do_asr=True,
        #     model_size="base" # "base" is a good balance of speed and accuracy
        # )
            
        )

        accelerator = AcceleratorOptions(
            num_threads=os.cpu_count(),
            device=AcceleratorDevice.CPU
        )

        # 2. Setup the Media Pipeline
        media_pipeline = AsrPipelineOptions(
            accelerator_options=accelerator,  # <--- Device setting goes here
            asr_options=asr_model_specs.WHISPER_BASE
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
                InputFormat.MD,
                InputFormat.AUDIO,     # ✅ NEW
                # InputFormat.VIDEO      # ✅ NEW
            ],
            format_options={
            # PDFs use the specialized PDF pipeline
            InputFormat.PDF: PdfFormatOption(pipeline_options=true_pdf_pipeline),
            
            # Word and PPTX use the standard options but can still use the same pipeline
            InputFormat.DOCX: WordFormatOption(pipeline_options=doc_pipeline),
            InputFormat.PPTX: PowerpointFormatOption(pipeline_options=doc_pipeline),
            
            # Images (JPG, PNG) also need this to trigger the Azure Vision description
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=doc_pipeline),
            InputFormat.HTML: HTMLFormatOption(pipeline_options=doc_pipeline),
            InputFormat.MD: MarkdownFormatOption(pipeline_options=doc_pipeline),
            
            # Excel & CSV
            InputFormat.XLSX: ExcelFormatOption(pipeline_options=doc_pipeline),

            # Auido and Video
            InputFormat.AUDIO: AudioFormatOption(
                    pipeline_cls=AsrPipeline, 
                    pipeline_options=media_pipeline
                ),
            }
        )

    def summarize_standalone_image(self, file_path):
        """
        Calls Azure Vision directly to get a high-level summary of a standalone image.
        """
        try:
            az_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            # Ensure your endpoint ends with the chat completions path
            azure_url = "https://newdocintel.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview"

            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            headers = {
                "Content-Type": "application/json",
                "api-key": az_api_key,
            }

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Identify what this image is (e.g., ID card, website screenshot, invoice). Provide a 2-sentence high-level summary of its content."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post(azure_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"⚠️ Could not generate standalone summary: {e}")
            return None

    def summarize_media_content(self, text_content, file_type):
        """Generates a high-level summary of a long transcript using Azure."""
        try:
            az_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_url = "https://newdocintel.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview"
            
            headers = {"Content-Type": "application/json", "api-key": az_api_key}
            payload = {
                "messages": [
                    {"role": "system", "content": f"You are a media analyst. Summarize this {file_type} transcript in 3 bullet points."},
                    {"role": "user", "content": text_content[:4000]} # Limit tokens
                ]
            }
            resp = requests.post(azure_url, headers=headers, json=payload)
            return resp.json()['choices'][0]['message']['content']
        except:
            return "No summary available."
        
    def extract_and_summarize_frames(self, video_path, doc_name, img_dir, interval_seconds=3):
        """Captures frames at intervals and generates a visual narrative."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: return "Could not analyze video frames."
        
        frame_interval = int(fps * interval_seconds)
        frame_count = 0
        timeline_entries = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_interval == 0:
                timestamp = int(frame_count / fps)
                frame_name = f"frame_{timestamp}s.png"
                frame_path = img_dir / frame_name
                cv2.imwrite(str(frame_path), frame)

                try:
                    result = self.converter.convert(str(frame_path))
                    # Use .get() or check if document has content to avoid errors
                    ocr_text = result.document.export_to_markdown() if result.document.texts else "[No text detected]"
                except Exception as e:
                    ocr_text = f"[OCR skipped: {e}]"
                
                # Use your existing Azure Vision summary function
                description = self.summarize_standalone_image(frame_path)
                timeline_entries.append(f"**[{timestamp}s]:** {description} \n\n Frame-OCR: {ocr_text} \n")
                
            frame_count += 1
        
        cap.release()
        return "\n".join(timeline_entries) if timeline_entries else "No visual activity detected."

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

        md_file = md_dir / f"{doc_name}.md"

        # Save markdown
        document.save_as_markdown(
            filename=md_file,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=img_dir,
            include_annotations=True
        )
        
        enrichment_header = ""
        ext = file_path.suffix.lower()
        if ext in ['.mp3', '.mp4', '.wav', '.mov', '.avi']:
            # Generate Global Summary
            media_summary = self.summarize_media_content(md_file, ext)
            header = f"## MEDIA SUMMARY ({ext.upper()})\n{media_summary}\n\n---\n\n"
            md_content = header

            if ext in ['.mp4', '.mov', '.avi']:
                print(f"🎬 Analyzing visual timeline for: {file_path.name}")
                visual_timeline = self.extract_and_summarize_frames(file_path, doc_name, img_dir)
                enrichment_header += f"## VISUAL TIMELINE\n{visual_timeline}\n\n"
            
            with open(md_file, "a", encoding="utf-8") as f:
                    f.write("\n\n"+md_content+"\n\n"+enrichment_header)


        if ext in ['.png', '.jpg', '.jpeg', '.bmp', ".gif"]:
            summary = self.summarize_standalone_image(file_path)
            if summary:
                # Injecting at the top so it's the first thing the RAG bot sees
                md_content = f"## IMAGE SUMMARY\n{summary}\n\n---\n\n" 

            # 3. Manually save the final enriched content to a file
            # md_file = md_dir / f"{doc_name}_enriched.md"
                with open(md_file, "a", encoding="utf-8") as f:
                    f.write("\n"+md_content+"\n")


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
