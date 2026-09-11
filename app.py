import os
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODELS = {
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "Qwen 3.6 27B": "qwen/qwen3.6-27b",
}

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 5


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(
        BytesIO(uploaded_file.getvalue())
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        # Clean extra whitespace
        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# ============================================================
# TOKENIZATION + CHUNKING
# ============================================================

def create_chunks(
    pages,
    tokenizer,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    for page in pages:

        text = page["text"]

        # Convert text into token IDs
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        start = 0

        while start < len(token_ids):

            end = min(
                start + chunk_size,
                len(token_ids)
            )

            chunk_token_ids = token_ids[start:end]

            # Convert tokens back to readable text
            chunk_text = tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                        "token_count": len(chunk_token_ids)
                    }
                )

            # Stop when we reach the end
            if end == len(token_ids):
                break

            # Move forward while keeping overlap
            start = end - overlap

    return chunks


# ============================================================
# CREATE FAISS VECTOR DATABASE
# ============================================================

def create_faiss_index(
    chunks,
    embedding_model
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    # Dimension of embedding vectors
    dimension = embeddings.shape[1]

    # Inner Product works as cosine similarity
    # because embeddings are normalized
    index = faiss.IndexFlatIP(dimension)

    # Store vectors in FAISS
    index.add(embeddings)

    return index


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    embedding_model,
    top_k=DEFAULT_TOP_K
):

    # Convert question into embedding
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Don't request more results than available
    k = min(
        top_k,
        index.ntotal
    )

    # Similarity search
    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        chunk = chunks[int(index_number)].copy()

        chunk["score"] = float(score)

        results.append(chunk)

    return results


# ============================================================
# GROQ LLM
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    model_id
):

    # Get API key from Streamlit Secrets
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    client = Groq(
        api_key=api_key
    )

    # Build context
    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a helpful Retrieval-Augmented Generation assistant.

Answer the user's question using ONLY the document context provided below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I couldn't find that information in the uploaded document."
3. Keep the answer clear and concise.
4. Mention page numbers when useful.
5. Do not invent information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
""".strip()

    response = client.chat.completions.create(

        model=model_id,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering "
                    "assistant. Use only the supplied context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=800
    )

    return response.choices[0].message.content


# ============================================================
# USER INTERFACE
# ============================================================

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)

st.caption(
    "PDF → Extraction → Token Chunking → "
    "Embeddings → FAISS → Retrieval → Groq"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose Groq Model",
    list(GROQ_MODELS.keys())
)

selected_model = GROQ_MODELS[
    selected_model_name
]

top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=2,
    max_value=8,
    value=DEFAULT_TOP_K
)


# ============================================================
# SESSION STATE
# ============================================================

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # Process only when a new PDF is uploaded
    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing your PDF..."
        ):

            # --------------------------------------------
            # STEP 1: Extract text
            # --------------------------------------------

            pages = extract_pdf_text(
                uploaded_file
            )

            if not pages:

                st.error(
                    "No selectable text was found "
                    "in this PDF."
                )

                st.stop()

            # --------------------------------------------
            # STEP 2: Load tokenizer
            # --------------------------------------------

            tokenizer = load_tokenizer()

            # --------------------------------------------
            # STEP 3: Tokenization + chunking
            # --------------------------------------------

            chunks = create_chunks(
                pages,
                tokenizer
            )

            if not chunks:

                st.error(
                    "No text chunks could be created."
                )

                st.stop()

            # --------------------------------------------
            # STEP 4: Load embedding model
            # --------------------------------------------

            embedding_model = (
                load_embedding_model()
            )

            # --------------------------------------------
            # STEP 5: Create FAISS index
            # --------------------------------------------

            faiss_index = create_faiss_index(
                chunks,
                embedding_model
            )

            # --------------------------------------------
            # STEP 6: Store in session state
            # --------------------------------------------

            st.session_state.faiss_index = (
                faiss_index
            )

            st.session_state.chunks = chunks

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.messages = []

        st.success(
            f"✅ PDF processed successfully!"
        )

        st.info(
            f"Pages: {len(pages)} | "
            f"Chunks: {len(chunks)}"
        )


# ============================================================
# CHAT SECTION
# ============================================================

if st.session_state.faiss_index is not None:

    st.divider()

    st.subheader(
        f"📖 {st.session_state.file_name}"
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # Chat input
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        # --------------------------------------------
        # Display user question
        # --------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # --------------------------------------------
        # Generate answer
        # --------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    embedding_model = (
                        load_embedding_model()
                    )

                    # Retrieve relevant chunks
                    retrieved_chunks = (
                        retrieve_relevant_chunks(
                            question,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            embedding_model,
                            top_k
                        )
                    )

                    # Generate answer
                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                        selected_model
                    )

                    st.markdown(answer)

                    # --------------------------------
                    # Show sources
                    # --------------------------------

                    with st.expander(
                        "🔎 Retrieved Sources"
                    ):

                        for number, chunk in enumerate(
                            retrieved_chunks,
                            start=1
                        ):

                            st.markdown(
                                f"**Source {number} — "
                                f"Page {chunk['page']} — "
                                f"Similarity: "
                                f"{chunk['score']:.3f}**"
                            )

                            st.write(
                                chunk["text"]
                            )

                    # Save assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as error:

                    st.error(
                        f"Something went wrong: {error}"
                                          )
