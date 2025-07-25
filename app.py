import os
import pandas as pd
import fitz  # PyMuPDF
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI  # ✅ Updated import

# ✅ Streamlit page settings
st.set_page_config(page_title="📊 Wheat Data RAG Chat", layout="centered")
st.title("📊 Wheat Data AI Chat App (Excel/PDF + Mistral)")

# ✅ Upload Excel or PDF
uploaded_file = st.file_uploader("Upload Excel (.xlsx) or PDF file", type=["xlsx", "pdf"])

if uploaded_file:
    text_data = []
    file_name = uploaded_file.name

    # ✅ Handle Excel
    if file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()  # Clean spaces from column names

        if 'BROKERS' in df.columns:
            df = df.dropna(subset=['BROKERS'])  # Keep rows with brokers

        for _, row in df.iterrows():
            row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
            text_data.append(row_text)

    # ✅ Handle PDF
    elif file_name.endswith(".pdf"):
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_data = [page.get_text() for page in pdf_doc]

    else:
        st.error("Only .xlsx and .pdf files are supported.")

    # ✅ Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=chunk) for text in text_data for chunk in splitter.split_text(text)]

    # ✅ Vector store with HuggingFace + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()

    # ✅ Mistral LLM via OpenRouter
    api_key = st.secrets["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"]  # Add your OpenRouter key to .streamlit/secrets.toml
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    llm = ChatOpenAI(
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.7
    )

    # ✅ Chat Interface
    question = st.text_input("Ask your question about the uploaded file")

    if st.button("Get Answer") and question:
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs[:3]])
        if len(context) > 2000:
            context = context[:2000]

        messages = [
            SystemMessage(content="Answer based only on the context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
        ]

        result = llm(messages)
        st.success(result.content)
