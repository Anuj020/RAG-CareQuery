import os
import streamlit as st
from dotenv import load_dotenv

from rag_retriever import RAGRetriever
from vectors import VectorStore
from embedding import EmbeddingManager
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(page_title="🩺 Symptom Checker", page_icon="🧠", layout="centered")

st.title("🧠 AI-Powered Symptom Checker")
st.write(
    "Ask any health-related question and get AI-generated answers powered by RAG (Retrieval-Augmented Generation)."
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@st.cache_resource
def load_llm():
    if not GOOGLE_API_KEY:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.8,
        api_key=GOOGLE_API_KEY,
    )


@st.cache_resource
def load_retriever(persist_directory: str = "vector_store/", collection_name: str = "Csv_data"):
    """
    Load existing persistent ChromaDB index.
    IMPORTANT: This must already be built OFFLINE (e.g., by CI or local run).
    """
    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore(collection_name=collection_name, persist_directory=persist_directory)
    return RAGRetriever(vectorstore, embedding_manager)


def rag(query, retriever, llm, top_k=3, min_score=0.1, return_context=True):
    """
    Same RAG logic as your main.py, but kept inside app.py to avoid importing main.py
    (which triggers offline ingestion imports).
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {"answer": "No relevant context found.", "sources": [], "confidence": 0.0, "context": ""}

    context = "\n\n".join([doc["content"] for doc in results])
    sources = [
        {
            "sources": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:300] + "...",
        }
        for doc in results
    ]
    confidence = max([doc["similarity_score"] for doc in results])

    prompt = (
        "Use the following context to answer the question concisely.\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)

    output = {"answer": response.content, "sources": sources, "confidence": confidence}
    if return_context:
        output["context"] = context
    return output


llm = load_llm()
if llm is None:
    st.error("GOOGLE_API_KEY is missing. Add it in Hugging Face Secrets or .env.")
    st.stop()

rag_retriever = load_retriever()

query = st.text_input("💬 Enter your question (e.g., What is Diabetes?)")

if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving information..."):
            result = rag(query, rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)

        st.subheader("🩺 Answer:")
        st.write(result["answer"])

        st.caption(f"Confidence: {result['confidence']:.2f}")

        st.markdown("### 📚 Sources:")
        for src in result["sources"]:
            st.markdown(f"- `{src['sources']}` (score: {src['score']:.3f})")

        with st.expander("🔍 View Retrieved Context"):
            st.text(result.get("context", ""))
