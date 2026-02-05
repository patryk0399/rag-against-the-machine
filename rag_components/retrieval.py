from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig, load_config


def _load_vector_store(cfg: AppConfig) -> FAISS:
    index_root = Path(cfg.data_dir) / "index" / "faiss"
    print("[query] Using index root:", index_root)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)
    print("[query] Initialising embeddings with model:", cfg.embedding_model_name)
    print("[query] embeddings: ", embeddings)

    vector_store = FAISS.load_local(
        folder_path=str(index_root),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    print("[query] Loaded vector store.")
    print("[query] Stored chunks:", len(vector_store.docstore._dict))  # type: ignore[attr-defined]

    return vector_store

def retrieve(
    query: str,
    k: int = 3,
    cfg: Optional[AppConfig] = None,
) -> List[Document]:

    if cfg is None:
        cfg = load_config()

    vector_store = _load_vector_store(cfg)
    docs = vector_store.similarity_search(query, k=k)

    print(f"[query] Retrieved {len(docs)} chunk(s).")
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "<unknown>")
        title = d.metadata.get("title", "<no title>")
        chunk_index = d.metadata.get("chunk_index", "?")
        print(f"[query]  [{i}] source={source} title={title} chunk_index={chunk_index}")

    print("DOCS1:\n", docs)
    return docs