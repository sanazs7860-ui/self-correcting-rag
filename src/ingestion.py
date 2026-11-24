import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from . import config

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore():
    return Chroma(
        collection_name="manuals",
        embedding_function=get_embeddings(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )


def ingest() -> int:
    """Read all PDFs in data/, chunk them, and rebuild the vector DB."""
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {config.DATA_DIR}")

    docs = []
    for pdf in pdfs:
        docs.extend(PyMuPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    if config.VECTOR_DB_DIR.exists():
        shutil.rmtree(config.VECTOR_DB_DIR)

    Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        collection_name="manuals",
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    return len(chunks)


if __name__ == "__main__":
    print(f"Ingested {ingest()} chunks.")
