from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(chunks, persist_directory: str = "chroma_db"):
    embeddings = create_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
