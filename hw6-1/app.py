# app.py
import streamlit as st
from rag import VectorRAG

st.set_page_config(page_title="HW6-1 Vector RAG", layout="wide")
st.title("HW6-1: Vector RAG (Streamlit)")

# Initialize RAG instance in session state to avoid repeated initialization
if "rag" not in st.session_state:
    st.session_state.rag = None

with st.sidebar:
    st.header("Indexing")
    data_dir = st.text_input("Data folder", value="data")
    k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=5)
    if st.button("Build / Rebuild Index"):
        with st.spinner("Initializing RAG and indexing documents..."):
            st.session_state.rag = VectorRAG()
            n = st.session_state.rag.index_folder(data_dir)
        st.success(f"Indexed {n} chunks from {data_dir}")

st.divider()

query = st.text_input("Ask a question", placeholder="Type your question here...")

if st.button("Search"):
    if st.session_state.rag is None:
        st.warning("Please build the index first by clicking 'Build / Rebuild Index' in the sidebar.")
    else:
        hits = st.session_state.rag.retrieve(query, k=k)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Answer (baseline)")
            st.write(st.session_state.rag.naive_answer(query, hits))

        with col2:
            st.subheader("Retrieved Sources")
            if not hits:
                st.info("No hits found.")
            for h in hits:
                st.markdown(f"**{h['source']} — chunk {h['chunk']}**  (distance={h['distance']:.4f})")
                st.code(h["text"][:900])