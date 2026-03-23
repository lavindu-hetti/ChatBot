import os

from langchain_community.document_loaders import PyPDFLoader


def save_uploaded_file(uploaded_file, data_dir: str = "data") -> str:
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()
