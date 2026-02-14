# 🧠 Case Intelligence: Multimodal RAG with Docling & Chroma

An advanced **Retrieval-Augmented Generation (RAG)** system designed for comprehensive case management.

This system processes **PDFs, Images, Audio, and Video** into a unified vector database using:

- **Docling** → Structured document parsing  
- **Whisper (ASR)** → Speech-to-text transcription  
- **Azure GPT-4 Vision** → Visual scene understanding + OCR  
- **ChromaDB** → Persistent hybrid vector database  

---

# 🛠️ Prerequisites

## 1️⃣ Python
- Install **Python 3.10 or 3.11**
- Verify:
```bash
python --version
```

---

## 2️⃣ FFmpeg (Required for Audio/Video Processing)

### ✅ Windows (Recommended)
```bash
winget install ffmpeg
```

### Alternative (Chocolatey)
```bash
choco install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

---

## 3️⃣ Install `uv` (Recommended Package Manager)

### Windows (PowerShell)
```bash
winget install AstralSh.uv
```

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:
```bash
uv --version
```

> You may use `pip`, but **uv is faster and recommended**.

---

# 📂 Project Structure

```
.
├── main.py              # Ingestion script (Uploads & Parses)
├── chat2.py             # RAG Chat interface
├── .env                 # API Keys and Secrets
├── parsers/
│   └── all_parser8.py   # Multimodal SmartDocumentParser
├── data/
│   ├── uploads/         # Drop raw files here
│   ├── output/          # Parsed Markdown, JSON, Images
│   └── chroma_db/       # Persistent Vector Database
└── README.md
```

---

# ⚙️ Setup & Installation

## 1️⃣ Navigate to Project

```bash
cd intel-3
```

---

## 2️⃣ Create `.env` File

Create a `.env` file in the root directory:

```env
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# Optional (for Hugging Face transcription fallback)
HF_TOKEN=your_token_here
```

---

## 3️⃣ Install Dependencies

### ✅ Using `uv` (Preferred)

```bash
git clone <repo>
cd intel-3
uv sync

```

If `uv sync` removes Whisper accidentally:

```bash
uv pip install openai-whisper
```

---

### Alternative: Using pip

```bash
pip install "docling[asr]" openai-whisper opencv-python chromadb python-dotenv requests langchain-huggingface
```

---

# 🚀 How to Use

---

# 1️⃣ Ingesting Documents (Upload Phase)

### Step 1 — Add Files

Place your files into:

```
data/input/
```

Supported formats:
- 📄 PDFs, Docs, xlsx
- 🎥 MP4 videos
- 🎙️ MP3 audio
- 🖼️ Images (JPG, PNG, etc.)

---

### Step 2 — Run Ingestion

```bash
uv run main.py
```

When prompted:

```
Enter Collection Name: case_123
```

This creates your **Case Room** in ChromaDB.

---

## 🔄 What Happens During Ingestion?

- PDFs → Parsed into structured Markdown + JSON
- Videos → Frames extracted every 4 seconds
- Azure Vision → Scene description + OCR text
- Audio → Whisper timestamped transcripts
- Everything → Embedded and stored in ChromaDB with metadata

---

# 2️⃣ Start the RAG Chat

Once ingestion is complete:

```bash
uv run chat2.py
```

You can now ask questions like:

- "What phone number appears on the sticker at the start of the video?"
- "Summarize the meeting audio."
- "What happens around timestamp 02:15?"
- "List all names mentioned in the transcript."

The assistant has access to:

- 📄 Parsed documents  
- 🎥 Video frame descriptions  
- 👁️ OCR extracted text  
- 🎙️ Timestamped transcripts  
- 🧠 Metadata (timestamps, filenames, source references)  

---

# 🔍 Advanced Features

## 🎥 Video Visual Timeline
- Extracts frames every 4 seconds
- Uses Azure GPT-4 Vision for:
  - Scene understanding
  - OCR on signs, labels, stickers
- Indexed with timestamps for precise retrieval

---

## 🎙️ Audio Intelligence
- Local Whisper via `docling[asr]`
- Timestamped transcripts
- Embedded for semantic search

---

## 🔎 Hybrid Search (ChromaDB)
- Vector similarity search
- Metadata filtering (filename, timestamp)
- Enables citation of exact video moments

---

# ⚠️ Troubleshooting

## ❌ WinError 2
FFmpeg is not in PATH.

Check:
```bash
ffmpeg -version
```

If not found, reinstall with:
```bash
winget install ffmpeg
```

---

## ❌ AttributeError (InputFormat)
Docling version issue. Upgrade:

```bash
uv pip install -U docling
```

---

## ❌ Hugging Face Connection Errors

After first successful model download, run offline mode:

### Windows
```powershell
$env:HF_HUB_OFFLINE=1
```

---

# 🧠 Architecture Overview

```
Multimodal Input (PDF / Video / Audio / Image)
            ↓
Docling / Whisper / Azure Vision
            ↓
Structured Content + Metadata
            ↓
Embeddings
            ↓
ChromaDB (Persistent Storage)
            ↓
RAG Chat Interface
```

---

# 📌 Recommended Workflow

1. Install prerequisites
2. Configure `.env`
3. Drop files into `data/uploads`
4. Run `main.py`
5. Start `chat2.py`
6. Investigate your case intelligently

---

# 🚀 Future Enhancements (Optional Ideas)

- Role-based case rooms
- Timeline visualization dashboard
- Cross-case linking
- Evidence scoring system
- Multi-user collaboration
- Docker deployment

---

# 🏁 You're Ready

Your **Multimodal Case Intelligence RAG system** is now ready for intelligent investigation workflows.

Happy Investigating 🔎
