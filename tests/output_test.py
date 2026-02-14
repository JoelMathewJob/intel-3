import os
from pathlib import Path
from parsers.pdf_parser import SmartPDFParser
from dotenv import load_dotenv

load_dotenv()

def run_test():
    # 1. Initialize the tool with your data/output directory
    output_dir = "data/output"
    parser = SmartPDFParser(output_dir=output_dir)
    
    # 2. Identify test files in your input folder
    input_folder = Path("data/input")
    pdf_files = list(input_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDFs found in {input_folder}. Please add a file to test.")
        return

    print(f"Starting test for {len(pdf_files)} file(s)...")

    for pdf_path in pdf_files:
        # 3. Process the file using the refined 'artifacts_dir' logic
        # This will create data/output/md/ and data/output/images/
        md_file = parser.process(str(pdf_path))
        
        # 4. Immediate Validation Checks
        print("\n--- Validation Report ---")
        if md_file.exists():
            print(f"✅ Markdown created at: {md_file}")
        else:
            print(f"❌ Markdown missing!")

        images_dir = Path(output_dir) / "images"
        if images_dir.exists() and any(images_dir.iterdir()):
            img_count = len(list(images_dir.glob("*.png")))
            print(f"✅ Images folder created with {img_count} files.")
        else:
            print(f"⚠️ No images found. (Check if your PDF actually contains pictures)")

if __name__ == "__main__":
    run_test()