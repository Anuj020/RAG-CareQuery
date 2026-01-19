# from main import rag

# from rag_retriever import RAGRetriever
# from langchain_google_genai import ChatGoogleGenerativeAI
# import os
# from dotenv import load_dotenv
# import streamlit as st
# from vectors import VectorStore
# from embedding import EmbeddingManager

# load_dotenv()
# # 4️⃣ Entry point
# GOOGLE_API_KEY =  os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",temperature=0.8, api_key = GOOGLE_API_KEY)

#         # --- Streamlit Page Config ---
# st.set_page_config(page_title="🩺 Symptom Checker", page_icon="🧠", layout="centered")

#     # --- Title and Description ---
# st.title("🧠 AI-Powered Symptom Checker")
# st.write("Ask any health-related question and get AI-generated answers powered by RAG (Retrieval-Augmented Generation).")

# query = st.text_input("💬 Enter your question (e.g., What is Diabetes?)")
    
#     # --- Button to Trigger Search ---
# if st.button("🔍 Get Answer"):
#     if not query.strip():
#         st.warning("Please enter a valid question.")
#     else:
#         with st.spinner("Retrieving information..."):
#             # Load existing persisted vector store (built offline)
#             st.write("CWD:", os.getcwd())
#             st.write("vector_store exists?", os.path.exists("vector_store/"))
#             if os.path.exists("vector_store/"):
#                 st.write("vector_store files:", os.listdir("vector_store/")[:20])
#             st.write("data exists?", os.path.exists("data/"))
#             if os.path.exists("data/"):
#                 st.write("data files:", os.listdir("data/")[:20])

#             embedding_manager = EmbeddingManager()
#             vectorstore = VectorStore(collection_name="Csv_data", persist_directory="/Users/anuj/Desktop/GenAI/rag/notebook/vector_store/")
#             rag_retriever = RAGRetriever(vectorstore, embedding_manager)
#             result = rag(query, rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)

#         # --- Display Results ---
#         st.subheader("🩺 Answer:")
#         st.write(result["answer"])

#         st.caption(f"Confidence: {result['confidence']:.2f}")

#         # --- Display Sources ---
#         st.markdown("### 📚 Sources:")
#         for src in result["sources"]:
#             st.markdown(f"- `{src['sources']}` (score: {src['score']:.3f})")

#         # --- Expandable Context ---
#         with st.expander("🔍 View Retrieved Context"):
#             st.text(result["context"])



import os
import tarfile
import boto3
from dotenv import load_dotenv
import streamlit as st

from main import rag
from rag_retriever import RAGRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from vectors import VectorStore
from embedding import EmbeddingManager

load_dotenv()

def ensure_vector_store():
    """
    Ensures the persisted Chroma directory (vector_store/) exists locally.
    If missing (e.g., fresh Hugging Face container), download a tar.gz from S3 and extract it.
    """
    if not os.path.exists("vector_store"):
        st.write("vector_store/ not found. Downloading from S3...")

        bucket = os.environ.get("S3_BUCKET_NAME")
        if not bucket:
            st.error("Missing env var: S3_BUCKET_NAME")
            st.stop()

        s3 = boto3.client("s3")
        s3.download_file(bucket, "chroma_index.tar.gz", "chroma_index.tar.gz")

        with tarfile.open("chroma_index.tar.gz", "r:gz") as tar:
            tar.extractall()

        st.write("Vector store ready.")
    else:
        st.write("Vector store already exists.")

# ✅ Make sure vector store exists BEFORE loading it
ensure_vector_store()

# --- LLM setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
    api_key=GOOGLE_API_KEY
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="🩺 Symptom Checker", page_icon="🧠", layout="centered")

# --- Title and Description ---
st.title("🧠 AI-Powered Symptom Checker")
st.write("Ask any health-related question and get AI-generated answers powered by RAG (Retrieval-Augmented Generation).")

query = st.text_input("💬 Enter your question (e.g., What is Diabetes?)")

if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving information..."):
            # Debug info
            st.write("CWD:", os.getcwd())
            st.write("vector_store exists?", os.path.exists("vector_store/"))
            if os.path.exists("vector_store/"):
                st.write("vector_store files:", os.listdir("vector_store/")[:20])

            # ✅ Use relative path inside HF container (NOT your laptop path)
            embedding_manager = EmbeddingManager()
            vectorstore = VectorStore(
                collection_name="Csv_data",
                persist_directory="vector_store/"
            )

            rag_retriever = RAGRetriever(vectorstore, embedding_manager)
            result = rag(query, rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)

        # --- Display Results ---
        st.subheader("🩺 Answer:")
        st.write(result["answer"])

        st.caption(f"Confidence: {result['confidence']:.2f}")

        # --- Display Sources ---
        st.markdown("### 📚 Sources:")
        for src in result["sources"]:
            st.markdown(f"- `{src['sources']}` (score: {src['score']:.3f})")

        # --- Expandable Context ---
        with st.expander("🔍 View Retrieved Context"):
            st.text(result["context"])

    

