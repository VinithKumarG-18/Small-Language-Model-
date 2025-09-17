import os
import pdfplumber
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----- ENV & CONFIG -----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found! Please set it as an environment variable.")
    st.stop()

PDF_PATH = os.getenv("PDF_PATH", "docs/VanDeGraaff.pdf")

# ----- GROQ CLIENT -----
client = Groq(api_key=GROQ_API_KEY)

# ----- SYSTEM PROMPT -----
system_prompt = """
[Role / Core]

You are a polite and helpful physics tutor. Your task is to explain concepts about the Van de Graaff generator in a clear, concise, and friendly manner, suitable for beginners and students. Always keep your explanations short and sweet, focusing on the key points without unnecessary detail.

Whenever possible, base your answers primarily on the information extracted from the provided PDF document about the Van de Graaff generator. Use that context to give precise and accurate explanations.

If the PDF content does not cover the query fully, provide a brief, general explanation grounded in standard physics knowledge, but keep it simple and to the point.

**VERY IMPORTANT:**
Whenever you include an equation, you must write it in proper LaTeX format, and always enclose it between double dollar signs like this:
$$ C = 4 \\pi \\varepsilon_0 \\frac{{r_1 r_2}}{{r_2 - r_1}} $$

This ensures it is rendered correctly in the chat interface.

Maintain a positive and respectful tone in every response, encouraging learning and curiosity.

PDF Content :{context_prompt}

[End Role]
"""

# ----- FUNCTIONS -----
def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text

def chunk_text(text, chunk_size=350):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return model, np.array(embeddings, dtype=np.float32)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_top_k(query, model, index, chunks, k=3):
    query_vec = model.encode([query]).astype(np.float32)
    D, I = index.search(query_vec, k)
    return "\n".join([chunks[i] for i in I[0] if i < len(chunks)])


# ----- STREAMLIT UI -----
st.set_page_config(page_title="Physics Tutor Chatbot", page_icon="⚡", layout="centered")
st.title("⚡ Van de Graaff Generator Tutor")
st.caption("Ask me anything about the Van de Graaff generator!")

@st.cache_resource
def load_index(pdf_path):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    model, embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    return model, index, chunks

try:
    model, index, chunks = load_index(PDF_PATH)
except FileNotFoundError:
    st.error(f"❌ PDF file not found at {PDF_PATH}. Please check the path.")
    st.stop()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if "$$" in content:
            parts = content.split("$$")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    st.latex(part.strip())
                elif part.strip():
                    st.markdown(part)
        else:
            st.markdown(content)

# --- User Input ---
if prompt := st.chat_input("Ask your question about Van de Graaff generator..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = retrieve_top_k(prompt, model, index, chunks)
    context_prompt = f"\n[Reference Material]\n{context}\n[End Reference]\n"
    combined_prompt = system_prompt.format(context_prompt=context_prompt) + "\n[User Query]\n" + prompt

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": combined_prompt}],
                    max_tokens=500,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"⚠️ Error: {e}"

            if "$$" in reply:
                parts = reply.split("$$")
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        st.latex(part.strip())
                    elif part.strip():
                        st.markdown(part)
            else:
                st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
