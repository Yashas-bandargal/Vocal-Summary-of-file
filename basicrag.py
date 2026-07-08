"""
=====================================================================
 RAG-based PDF Question Answering Assistant (with Voice Output)
=====================================================================

WHAT IS THIS PROJECT?
----------------------
This is a beginner-friendly implementation of RAG (Retrieval-Augmented
Generation). RAG is a technique where instead of asking an LLM a
question directly (and hoping it "remembers" the right facts), we:

  1. First SEARCH our own documents for the most relevant pieces of
     text related to the question (this is the "Retrieval" part).
  2. Then we GIVE that retrieved text to the LLM as context and ask
     it to answer USING that context (this is the "Generation" part).

This is powerful because:
  - The LLM can answer questions about documents it has never been
    trained on (your own PDFs).
  - It reduces hallucination, because the model is grounded in real
    text instead of guessing from memory.
  - It scales to large documents that would never fit in a single
    prompt.

HOW TO RUN THIS PROJECT (short version, full details at the bottom):
  1. pip install the required libraries (see requirements below).
  2. Get a free Gemini API key from https://aistudio.google.com/apikey
  3. Run:  streamlit run app.py
  4. Upload PDFs, enter your API key, ask questions, and listen to
     the spoken answer.

=====================================================================
 REQUIRED LIBRARIES AND WHY WE USE THEM
=====================================================================
- streamlit            -> Quick way to build a web UI in pure Python.
- pypdf                -> Reads PDF files and extracts raw text.
- sentence-transformers -> Converts text chunks into embeddings
                           (numeric vectors that capture MEANING).
- faiss-cpu            -> A vector database/index used to store
                           embeddings and perform fast semantic
                           (meaning-based) search.
- google-generativeai   -> Free-tier LLM (Gemini) used to generate
                           the final natural-language answer.
- gTTS                  -> Converts the final text answer into an
                           audio (speech) file (Text-to-Speech).

INSTALL COMMAND (also shown at the end of this file):
    pip install streamlit pypdf sentence-transformers faiss-cpu google-generativeai gTTS numpy
=====================================================================
"""

import os
import tempfile

import numpy as np
import streamlit as st
from pypdf import PdfReader                      #= For reading PDF files
from sentence_transformers import SentenceTransformer  # For embeddings
import faiss                                      # For vector search
from gtts import gTTS                             # For text-to-speech
import google.generativeai as genai               # Free LLM (Gemini)


# =====================================================================
# STEP 1: EXTRACT TEXT FROM PDFs
# =====================================================================
# WHY THIS IS NEEDED:
# PDFs are not plain text - they store text along with layout/formatting
# info. Before we can search or embed the content, we need to pull out
# the raw text so it can be processed like normal strings.
def extract_text_from_pdfs(uploaded_files):
    """Reads one or more uploaded PDF files and returns their combined text."""
    all_text = ""
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # Some pages (e.g. scanned images) may return None
                    all_text += page_text + "\n"
        except Exception as e:
            st.error(f"Could not read file '{uploaded_file.name}': {e}")
    return all_text


# =====================================================================
# STEP 2: SPLIT TEXT INTO CHUNKS
# =====================================================================
# WHY CHUNKING IS NEEDED:
# A whole PDF can be thousands of words long - far too much to embed
# as a single vector (a single vector would blur together too many
# different ideas, making search inaccurate) or to feed directly to an
# LLM (context length limits). So we break the text into small,
# overlapping "chunks" (e.g. ~500 characters each). Each chunk covers
# one small topic, which makes semantic search much more precise.
#
# The "overlap" between chunks avoids cutting a sentence's meaning
# exactly at a chunk boundary and losing context.
def chunk_text(text, chunk_size=800, overlap=100):
    """Splits a long string into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but overlap a bit
    return chunks


# =====================================================================
# STEP 3: CONVERT CHUNKS INTO EMBEDDINGS
# =====================================================================
# WHAT ARE EMBEDDINGS?
# An embedding is a list of numbers (a vector) that represents the
# MEANING of a piece of text. Text with similar meaning will have
# vectors that are close together in this numeric space, even if the
# exact words used are different. This is what allows "semantic"
# search (searching by meaning) instead of plain keyword matching.
#
# We use a pre-trained sentence-transformer model ("all-MiniLM-L6-v2")
# which is small, fast, free, and runs locally on CPU.
@st.cache_resource  # Cache the model so it loads only once, not on every rerun
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_embeddings(chunks, model):
    """Converts a list of text chunks into a matrix of embedding vectors."""
    embeddings = model.encode(chunks, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


# =====================================================================
# STEP 4: STORE EMBEDDINGS IN A VECTOR INDEX (FAISS)
# =====================================================================
# WHY FAISS / A VECTOR DATABASE IS NEEDED:
# Once we have hundreds of chunk embeddings, we need a fast way to find
# the ones closest (most similar in meaning) to a new question's
# embedding. Comparing manually one by one would be slow at scale.
# FAISS (Facebook AI Similarity Search) is a library built exactly for
# this: it indexes vectors and lets us search "nearest neighbours"
# extremely quickly, even with millions of vectors.
def build_faiss_index(embeddings):
    """Builds a FAISS index from an array of embeddings for fast similarity search."""
    dimension = embeddings.shape[1]           # size of each embedding vector
    index = faiss.IndexFlatL2(dimension)      # simple index using L2 (Euclidean) distance
    index.add(embeddings)                     # add all chunk vectors to the index
    return index


# =====================================================================
# STEP 5: SEMANTIC SEARCH - RETRIEVE RELEVANT CHUNKS
# =====================================================================
# WHAT IS SEMANTIC SEARCH?
# Instead of matching exact keywords, we convert the user's QUESTION
# into an embedding too, then ask FAISS: "which chunk embeddings are
# closest to this question's embedding?". The closest chunks are the
# ones most likely to contain the answer, even if they don't share the
# exact same words as the question.
def semantic_search(question, model, index, chunks, top_k=3):
    """Returns the top_k chunks most semantically similar to the question."""
    question_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return retrieved_chunks


# =====================================================================
# STEP 6: GENERATE THE ANSWER USING AN LLM (RAG - Generation step)
# =====================================================================
# HOW RAG WORKS (put together):
#   Question -> embed question -> FAISS search -> retrieve top chunks
#   -> build a prompt that says "using ONLY this context, answer the
#   question" -> send to LLM -> LLM generates a grounded answer.
#
# We use Google's Gemini free-tier API here because it's free to get
# an API key and easy to call. You could swap this function for any
# other LLM (OpenAI, local Ollama model, etc.) without changing the
# rest of the RAG pipeline - that's the beauty of RAG's design.
def generate_answer(question, context_chunks, api_key):
    """Sends the retrieved context + question to the LLM and returns its answer."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        context_text = "\n\n---\n\n".join(context_chunks)

        prompt = f"""You are a helpful assistant. Answer the question using ONLY
the context provided below. If the answer is not present in the context,
say "I could not find this information in the uploaded PDF(s)."

Context:
{context_text}

Question: {question}

Answer:"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error while generating answer: {e}"


# =====================================================================
# STEP 7: TEXT-TO-SPEECH - SPEAK THE ANSWER ALOUD
# =====================================================================
# HOW TEXT-TO-SPEECH WORKS:
# gTTS (Google Text-to-Speech) sends the text to Google's TTS service,
# which returns an audio (MP3) file of that text being spoken. We save
# it to a temporary file and play it back inside the Streamlit app
# using st.audio().
def text_to_speech(text):
    """Converts text into an audio file and returns the file path."""
    try:
        tts = gTTS(text=text, lang="en")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech failed: {e}")
        return None


# =====================================================================
# STREAMLIT UI - MAIN APPLICATION
# =====================================================================
def main():
    st.set_page_config(page_title="RAG PDF Q&A Assistant", page_icon="📄")
    st.title("📄 RAG-based PDF Question Answering Assistant")
    st.write(
        "Upload one or more PDFs, ask a question, and get an AI-generated "
        "answer grounded in your documents — spoken aloud too!"
    )

    # ---- Sidebar: API key input ----
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey",
    )

    # ---- Step 1: Upload PDFs ----
    uploaded_files = st.file_uploader(
        "Upload PDF file(s)", type=["pdf"], accept_multiple_files=True
    )

    # We store processed data in Streamlit's session_state so it persists
    # between reruns (Streamlit reruns the whole script on every interaction).
    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.chunks = None

    if uploaded_files:
        if st.button("Process PDFs"):
            with st.spinner("Extracting text from PDFs..."):
                raw_text = extract_text_from_pdfs(uploaded_files)

            if not raw_text.strip():
                st.error("No readable text found in the uploaded PDF(s).")
            else:
                with st.spinner("Splitting text into chunks..."):
                    chunks = chunk_text(raw_text)

                with st.spinner("Creating embeddings (this may take a moment)..."):
                    embed_model = load_embedding_model()
                    embeddings = build_embeddings(chunks, embed_model)

                with st.spinner("Building FAISS vector index..."):
                    index = build_faiss_index(embeddings)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.success(f"Done! Processed {len(chunks)} chunks from your PDF(s).")

    # ---- Step 2: Ask a question ----
    if st.session_state.get("index") is not None:
        st.subheader("Ask a question about your PDF(s)")
        question = st.text_input("Your question")

        if st.button("Get Answer") and question:
            if not api_key:
                st.error("Please enter your Gemini API key in the sidebar first.")
            else:
                embed_model = load_embedding_model()

                with st.spinner("Searching for relevant content (semantic search)..."):
                    retrieved_chunks = semantic_search(
                        question, embed_model, st.session_state.index, st.session_state.chunks
                    )

                with st.spinner("Generating answer using the LLM..."):
                    answer = generate_answer(question, retrieved_chunks, api_key)

                st.subheader("✅ Answer")
                st.write(answer)

                with st.expander("🔍 Show retrieved source chunks (used as context)"):
                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(chunk)

                with st.spinner("Converting answer to speech..."):
                    audio_path = text_to_speech(answer)

                if audio_path:
                    st.subheader("🔊 Listen to the Answer")
                    st.audio(audio_path, format="audio/mp3")
                    os.remove(audio_path)  # clean up temp file after playing
    else:
        st.info("Upload PDF(s) and click 'Process PDFs' to get started.")


if __name__ == "__main__":
    main()
