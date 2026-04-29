import os
from dotenv import load_dotenv
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="QueryCraft", page_icon="🔍", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #13131a;
    --border:    #2a2a3a;
    --accent:    #7c6aff;
    --accent2:   #ff6ab0;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* hide default streamlit header */
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero title ── */
.qc-hero {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    margin: 0;
}
.qc-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* ── Divider ── */
.qc-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ── Success / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.6rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,255,0.2) !important;
}

/* ── Radio ── */
.stRadio label { color: var(--text) !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="qc-hero" style="font-size:1.8rem!important">🔍 QueryCraft</p>', unsafe_allow_html=True)
    st.markdown('<hr class="qc-divider">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📎 Upload your PDF", type="pdf")

    st.markdown('<hr class="qc-divider">', unsafe_allow_html=True)
    mode = st.radio("Choose Mode", ["💬 Q&A Mode", "🧠 Quiz Mode"])

    st.markdown('<hr class="qc-divider">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
    '<p style="color:#6b6b80;font-size:0.75rem;margin-top:2rem">Made by Nandini Bhatt</p>',unsafe_allow_html=True
)

# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="qc-hero">QueryCraft</p>', unsafe_allow_html=True)
st.markdown('<p class="qc-sub">Your AI-powered PDF intelligence engine</p>', unsafe_allow_html=True)
st.markdown('<hr class="qc-divider">', unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div style="
        background: #13131a;
        border: 1px solid #2a2a3a;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin-top: 2rem;
    ">
        <div style="font-size:3rem">📄</div>
        <h3 style="font-family:Syne,sans-serif;color:#e8e8f0;margin:1rem 0 0.5rem">Upload a PDF to begin</h3>
        <p style="color:#6b6b80">Use the sidebar to upload any PDF — then ask questions or generate a quiz!</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.success(f"✅ **{uploaded_file.name}** loaded — {len(chunks)} chunks ready!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── QUIZ MODE ──
    if "🧠 Quiz Mode" in mode:
        st.markdown("### 🧠 Quiz Mode")
        st.markdown('<p style="color:#6b6b80">AI will generate questions from your PDF — test your knowledge!</p>', unsafe_allow_html=True)

        if st.button("✨ Generate Quiz"):
            with st.spinner("Crafting questions from your PDF..."):
                quiz_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a teacher. Generate 5 multiple choice questions based on the given text. Format each question clearly with 4 options A, B, C, D and mention the correct answer at the end."},
                        {"role": "user", "content": f"Generate 5 quiz questions from this text:\n\n{text[:3000]}"}
                    ]
                )
                st.session_state.quiz = quiz_response.choices[0].message.content

        if "quiz" in st.session_state:
            st.markdown(f"""
            <div style="background:#13131a;border:1px solid #2a2a3a;border-radius:14px;padding:1.5rem;margin:1rem 0">
            {st.session_state.quiz}
            </div>
            """, unsafe_allow_html=True)

            user_answer = st.text_area("📝 Your answers (e.g. 1-A, 2-B, 3-C, 4-D, 5-A)")
            if st.button("🎯 Submit Answers"):
                with st.spinner("Evaluating your answers..."):
                    check_response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "You are a teacher. Check the student's answers and give score and detailed feedback."},
                            {"role": "user", "content": f"Quiz:\n{st.session_state.quiz}\n\nStudent answers:\n{user_answer}\n\nGive score out of 5 and feedback."}
                        ]
                    )
                    st.markdown("### 📊 Your Results")
                    st.markdown(check_response.choices[0].message.content)

    # ── Q&A MODE ──
    else:
        st.markdown("### 💬 Chat with your PDF")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask anything about your PDF...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            question_embedding = model.encode([question])
            D, I = index.search(np.array(question_embedding), k=2)
            relevant_text = " ".join([chunks[i] for i in I[0]])

            history = [{"role": "system", "content": "Answer the question based on the given PDF context only. If answer is not in context, say 'I could not find this in the PDF'."}]
            for msg in st.session_state.messages[-6:]:
                history.append({"role": msg["role"], "content": msg["content"]})
            history.append({"role": "user", "content": f"Context: {relevant_text}\n\nQuestion: {question}"})

            with st.spinner("QueryCraft is thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=history
                )
            answer = response.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)