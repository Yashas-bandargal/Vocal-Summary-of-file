# Vocal Summary of File

A mini Python project that transforms PDF documents into a searchable voice-enabled assistant. It extracts text from PDFs, creates vector embeddings using SentenceTransformers and FAISS, answers user questions using Gemini AI, and can play spoken responses via gTTS.

## What this project does

- Reads one or more PDF files and extracts selectable text.
- Splits document text into overlapping chunks for semantic retrieval.
- Uses `sentence-transformers` to build dense embeddings.
- Stores embeddings in a FAISS vector index.
- Uses semantic search to find the most relevant document chunks for each question.
- Generates answers with Google Gemini via the `google-genai` client.
- Optionally reads the answer aloud using `gTTS` and `playsound`.

## Requirements

- Python 3.10+ (tested with a modern Python 3 version)
- `faiss-cpu`
- `numpy`
- `pypdf`
- `gtts`
- `playsound`
- `sentence-transformers`
- `google-genai`

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the `basicrag.py` file in a working folder.

## Usage

1. Run the script:

```bash
python basicrag.py
```

2. Enter your Gemini API key when prompted.
3. Enter one or more PDF file paths separated by commas.
4. Ask questions about the PDF content.
5. Optionally choose to play the answer as spoken audio.

## Example

```bash
python basicrag.py
```

- Enter API key: `YOUR_GEMINI_API_KEY`
- Enter PDF path: `C:\Users\Asus\Desktop\example.pdf`
- Question: `What does this document describe?`
- Play answer as speech: `y`

## How it works

1. **PDF extraction**: `pypdf` reads each page and extracts text.
2. **Text chunking**: The script splits extracted text into overlapping sections to preserve context.
3. **Embedding generation**: `SentenceTransformer('all-MiniLM-L6-v2')` converts chunks into dense vector embeddings.
4. **Vector index**: FAISS stores and indexes embeddings for fast similarity search.
5. **Search**: The user question is embedded and compared against stored chunks.
6. **Answer generation**: Gemini generates a response using only the selected document context.
7. **Speech**: If requested, `gTTS` converts the text answer to speech and `playsound` plays it.

## Notes

- Ensure the PDF contains selectable text; scanned pages may not work without OCR.
- Keep your Gemini API key secure and do not commit it to version control.
- The script currently uses a single model name: `gemini-2.5-flash`.
