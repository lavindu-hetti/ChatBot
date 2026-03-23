from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_text_preview(docs, preview_length: int = 2000) -> str:
    extracted_text = "\n\n".join(doc.page_content for doc in docs)
    return extracted_text[:preview_length]


def chunk_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
