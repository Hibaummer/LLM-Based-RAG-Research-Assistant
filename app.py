# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:58:46 2026

@author: USER-ID
"""



import os
import time
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
import pymupdf

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import errors


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LLM RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 LLM RAG Research Assistant")
st.write(
    "Ask questions about your research paper "
    "and explore the passages supporting each answer."
)

# Use the folder containing app.py
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Research Paper.pdf"
INDEX_PATH = BASE_DIR / "research_paper.index"


# ==========================================
# 2. CHECK GEMINI API KEY
# ==========================================
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key not found. Set the GEMINI_API_KEY "
        "environment variable in CMD and restart the app."
    )
    st.stop()

client = genai.Client(api_key=api_key)


# ==========================================
# 3. CHECK PROJECT FILES
# ==========================================
if not PDF_PATH.exists():
    st.error(f"PDF file not found: {PDF_PATH.name}")
    st.stop()

if not INDEX_PATH.exists():
    st.error(f"FAISS index not found: {INDEX_PATH.name}")
    st.stop()


# ==========================================
# 4. LOAD FAISS INDEX AND EMBEDDING MODEL
# ==========================================
@st.cache_resource
def load_resources():
    faiss_index = faiss.read_index(str(INDEX_PATH))

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return faiss_index, embedding_model


# ==========================================
# 5. LOAD PDF AND RECREATE CHUNKS
# ==========================================
@st.cache_data
def load_chunks():
    doc = pymupdf.open(str(PDF_PATH))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    try:
        for page_num, page in enumerate(doc):
            page_text = page.get_text()

            page_chunks = splitter.split_text(page_text)

            for chunk in page_chunks:
                all_chunks.append({
                    "page": page_num + 1,
                    "text": chunk
                })
    finally:
        doc.close()

    return all_chunks


with st.spinner("Loading research paper and AI model..."):
    index, model = load_resources()
    chunks = load_chunks()


# ==========================================
# 6. VALIDATE THE INDEX
# ==========================================
if index.ntotal != len(chunks):
    st.error(
        f"Index mismatch: FAISS contains {index.ntotal} vectors, "
        f"but the PDF produced {len(chunks)} chunks. "
        "Rebuild the FAISS index using the same PDF and "
        "chunking settings before continuing."
    )
    st.stop()

if index.ntotal == 0:
    st.error("The FAISS index is empty.")
    st.stop()


# ==========================================
# 7. RETRIEVE RELEVANT PASSAGES
# ==========================================
def retrieve_chunks(question, k=3):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    # Match the cosine-similarity index used during indexing
    faiss.normalize_L2(query_embedding)

    k = min(k, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "page": chunks[idx]["page"],
                "text": chunks[idx]["text"],
                "score": float(score)
            })

    return results


# ==========================================
# 8. GENERATE ANSWER WITH GEMINI
# ==========================================
def generate_with_fallback(prompt):
    model_names = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]

    last_error = None

    for model_name in model_names:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                return response

            except errors.APIError as exc:
                last_error = exc
                status_code = getattr(exc, "code", None)

                # Retry temporary server or rate-limit errors
                temporary_errors = {429, 500, 502, 503, 504}

                if status_code not in temporary_errors:
                    raise RuntimeError(
                        f"Gemini API error ({status_code}): {exc}"
                    ) from exc

                if attempt == 0:
                    time.sleep(2)

                # Otherwise try the fallback model

    raise RuntimeError(
        "Gemini is temporarily unavailable or the request limit "
        "has been reached. Please wait and try again. "
        f"Last error: {last_error}"
    )


def answer_question(question, results):
    if not results:
        return "No relevant passages were found in the paper."

    context = "\n\n".join(
        f"[Source: Page {item['page']}]\n{item['text']}"
        for item in results
    )

    prompt = f"""
You are an assistant that answers questions about research papers.

Use only the research paper context provided below.

Instructions:
1. Answer in clear, beginner-friendly language.
2. Explain technical terms when helpful.
3. Do not invent facts or use unsupported outside information.
4. If the context does not answer the question, say so.
5. Include page references for important claims.
6. Distinguish information stated in the paper from your own inference.

Research paper context:
{context}

Question:
{question}

Answer:
"""

    response = generate_with_fallback(prompt)

    return response.text or "No answer was generated."


# ==========================================
# 9. STREAMLIT USER INTERFACE
# ==========================================
st.success(
    f"Research paper loaded successfully: "
    f"{len(chunks)} text chunks available."
)

question = st.text_input(
    "Ask a question about your research paper",
    placeholder="Why does the Transformer use attention instead of RNNs?"
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Retrieving passages and generating your answer..."
            ):
                results = retrieve_chunks(question, k=3)
                answer = answer_question(question, results)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View retrieved sources"):
                for i, item in enumerate(results, start=1):
                    st.markdown(
                        f"**Source {i} — Page {item['page']}**"
                    )

                    st.caption(
                        f"Similarity score: {item['score']:.3f}"
                    )

                    st.write(item["text"])
                    st.divider()

        except RuntimeError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(
                "An unexpected error occurred. "
                "Check your CMD terminal for details."
            )
            print(f"Unexpected error: {exc}")