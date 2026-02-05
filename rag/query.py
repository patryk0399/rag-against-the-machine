from __future__ import annotations
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
from agents.prompts import GENERAL_SYSTEM_PROMPT
from langchain_core.prompts.base import format_document
from langchain_core.prompts import PromptTemplate

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

"""RAG query helpers.

Prodiving functions that can be reused by other layers.
This module belongs to the context and LLM layers:
- Context layer:
  - Loads FAISS index built in rag.build_index.
  - Retrieves relevant chunks for a given query.
- LLM layer:
  - Builds the RAG-prompt from retrieved chunks with user query.
  - Uses the configured local LLM backend via get_local_llm.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig, load_config
from src.llm_backend import get_local_llm

def clean_docs(documents):
    doc_prompt = PromptTemplate.from_template(
    "Source: {source}\n"
    "Chunk: {chunk_index}\n"
    "{page_content}"
    )
    formatted_docs = []
    for d in documents:
        # If some metadata keys are missing (e.g., page/section/chunk), you can normalize upstream
        formatted_docs.append(format_document(d, doc_prompt))

    context_block = "\n\n---\n\n".join(formatted_docs)
    return context_block


def clean_docs2(documents):
    doc_prompt = PromptTemplate.from_template(
    # "Source: {source}\n"
    # "Page/Section: {page}{section}\n"
    # "Chunk: {chunk}\n"
    "{page_content}"
    )

    formatted_docs = []
    for d in documents:
        # If some metadata keys are missing (e.g., page/section/chunk), you can normalize upstream
        formatted_docs.append(format_document(d, doc_prompt))
    context_block = "\n".join(formatted_docs)
    return context_block

def _load_vector_store(cfg: AppConfig) -> FAISS:
    """Load the FAISS vector store for the current configuration.

    The index should to be under:
        data/index/faiss/ (faiss for now)#todo

    retrieve() and answer() can be used for debugging 
    (to test RAG only).
    """
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

    print("[query] Loaded FAISS vector store.")
    print("[query] Stored chunks:", len(vector_store.docstore._dict))  # type: ignore[attr-defined]

    return vector_store


def retrieve(
    query: str,
    k: int = 3,
    cfg: Optional[AppConfig] = None,
) -> List[Document]:
    """Retrieve the top-k documents for an user query.

    Parameters
    ----------
    query:
        User query String.
    k:
        Number of chunks to retrieve from the index.
    cfg:
        Optional AppConfig. If omitted, load_config() is used.

    Returns
    -------
    List[Document]
        Retrieved LangChain Document objects with metadata and content.
    """
    if cfg is None:
        cfg = load_config()
    print("[query] Loaded AppConfig from environment.")

    print(f"[query] Retrieving top-{k} chunks for query:")
    print("        ", repr(query))


    vector_store = _load_vector_store(cfg)
    docs = vector_store.similarity_search(query, k=k)

    # print(f"[query] Retrieved {len(docs)} chunk(s).")
    # for i, d in enumerate(docs, start=1):
    #     source = d.metadata.get("source", "<unknown>")
    #     title = d.metadata.get("title", "<no title>")
    #     chunk_index = d.metadata.get("chunk_index", "?")
    #     print(f"[query]  [{i}] source={source} title={title} chunk_index={chunk_index}")
    clean_d = clean_docs(docs)

    print("DOCS CLEAN?:\n", clean_d)
    return docs

class Retrieve2Input(BaseModel):
    query: str = Field(
        #default = None,
        #query: str = Field(..., min_length=1, description="...")
        description=
        """Input query for the retrival task. Must not be empty.
        Will be used for RAG functionality. 
        Based on this input query relevant documents are retrieved"""
                    )
    k: int = Field(
        default = 3,
        description = 
        """Number of chunks to retrieve from the index. Must be at least 3. At most 5.
        Range: 3 - 5.
        If the problem is simple, lower values for k suffice.
        If the problem is complex, higher values should be chosen for k.
        """
    )
    # Having to adjust this using natural language is tiring. The LLM does what it wants.
    # Needs to understand how much context is available to not overshoot the token
    # limit with just the retrieval.

@tool(args_schema=Retrieve2Input)
def retrieve2(
    query: str,
    k: int = 3,
) -> List[Document]:
    """Retrieve the top documents for an user query.

    Parameters
    ----------
    query:
        User query String.
    k:
        Number of chunks to retrieve from the index.
    Returns
    -------
    List[Document]
        Retrieved LangChain Document objects with metadata and content.
    """
    # if(query == None):
    #     return "No query given."

    # NOTE: right now we trust the tool calling mechanism of the selected LLM
    #       to give a valid query into the tool function call. 
    #       So the user query should land here via the LLM tool calling mechanism.
    #       From here the given query can be optimised for retrieval.
    #       Another possibility is it to optimise it before it is given to this
    #       function. Test what works better!

    cfg = load_config()
    print("[query] Loaded AppConfig from environment.")
    print("[query] The retrieve2 query is: ", query)
    llm = get_local_llm(cfg=cfg)
    
    content = f"""You are a RAG expert. 
                Your expertise lies in creating perfect queries for retrieval tasks.
                The queries should be optimised for document retrieval in the chemical industry domain.
                Optimise for one-shot retrieval. Your output must only contain the optimised query. DO NOT include any other
                information other than the optimised query.
                Here is the query you need to optimise: {query}.
                """
    better_query_message = SystemMessage(content=content)
    print("better_query_message: ", better_query_message.content)
    optimised_query = llm.invoke([better_query_message])
 
    print("[query] The retrieve2 optimiseed query is: ", optimised_query.content)

    # failsafe if tool calling mechanism of LLM fails to give proper k-values.
    # Despite a schema and natural language description of values to set
    # most models will still fail to set them correctly
    if(k<cfg.retrieve_k_min):
        print(f"query] k is lower than config min of {cfg.retrieve_k_min}. k is {k}")
        k=cfg.retrieve_k_min
        print(f"query] k was set via config to {cfg.retrieve_k_min}")
    print(f"[query] Retrieving top-{k} chunks for query:")
    print("        ", repr(query))


    vector_store = _load_vector_store(cfg)
    docs = vector_store.similarity_search(optimised_query.content, k=k)

    context_blocks: list[str] = []  
    for i, d in enumerate(docs, start=1): 
        source = d.metadata.get("source", "<unknown>")  
        chunk_index = d.metadata.get("chunk_index", "?")
        header = f"[{i}] Source: {source} | Chunk: {chunk_index}"
        context_blocks.append(f"{header}\n{d.page_content}")

    context_text = "\n\n".join(context_blocks)
    context_text = f"Tool output: {context_text}" # Why does it work more reliable when including this ????????????
                                                  # May need to structure the responses more literally 

    print("DOCS2:\n", context_text)
    return context_text

    # ----------------
    # context_blocks: list[str] = []  
    # for i, d in enumerate(docs, start=1): 
    #     source = d.metadata.get("source", "<unknown>")  
    #     #chunk_index = d.metadata.get("chunk_index", "?")
    #     header = f"[{i}]" # Source: {source}"
    #     context_blocks.append(f"{header}\n{d.page_content}")

    # context_text = "\n\n".join(context_blocks)
    # print("---------------------- context text: ", context_text)
    # return context_text



### potentially useful for "clean docs"/formatting
def _build_rag_prompt(query: str, docs: List[Document]) -> str:
    """Build a RAG prompt from the user query and retrieved documents."""
    context_blocks: list[str] = []  
    for i, d in enumerate(docs, start=1): 
        source = d.metadata.get("source", "<unknown>")  
        chunk_index = d.metadata.get("chunk_index", "?")
        header = f"[{i}] Source: {source} | Chunk: {chunk_index}"
        context_blocks.append(f"{header}\n{d.page_content}")

    context_text = "\n\n".join(context_blocks)
    print("[query]------------------------------------\n", context_text)
    system_prompt = GENERAL_SYSTEM_PROMPT

    user_prompt = f"""\
        Kontext:
        {context_text}

        Frage:
        {query}
        """

    llm_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return llm_prompt


def answer(
    query: str,
    k: int = 3,
    #cfg: Optional[AppConfig] = None,
) -> str:
    """Answer user prompt using RAG with the configured LLM backend.

    - Loads configuration (if not provided).
    - Loads FAISS vector store.
    - Retrieves top-k relevant chunks for the query.
    - Builds a RAG prompt.
    - Invokes the configured local LLM backend with the prompt.
    - Returns the model's response as a string.

    Parameters
    ----------  
    query:
        User query string. 
    k:
        Number of chunks to retrieve and include as context.
    cfg:
        Optional AppConfig. If omitted, load_config() is used.

    Returns
    -------
    str
        The answer produced by the selected LLM backend.
    """
    #if cfg is None:
    cfg = load_config()
    print("[query] Loaded AppConfig from environment.")

    print("[query] Using LLM backend:")#, cfg.llm_backend)
    docs = retrieve(query=query, k=k, cfg=cfg)

    if not docs:
        print("[query] No documents retrieved. Returning fallback message.")
        return "No relevant context found in the index. Unable to answer."

    prompt = _build_rag_prompt(query, docs)
    print("[query] Built RAG prompt. Prompt length (chars):", len(prompt))

    llm = get_local_llm(cfg)
    print("[query] Invoking LLM backend...")

    result = llm.invoke(prompt)

    # Convert result to string in case the backend returns a non-string object.
    answer_text = str(result)
    print("[query] LLM invocation completed. Answer length (chars):", len(answer_text))

    return answer_text


def main() -> None:
    """ CLI for manual RAG testing.   
    - python -m rag.query
      (then type the request when prompted)
    """
    # import sys  

    # if len(sys.argv) > 1:
    #     user_query = " ".join(sys.argv[1:])
    # else:
    #     user_query = input("Enter query: ").strip()
 
    # if not user_query:
    #     print("[query] Empty query provided. Exiting.")
    #     return

    # print("[query] Question:", repr(user_query))
    # answer_text = answer(user_query)
    # print("\n=== ANSWER ===")
    # print(answer_text) 
    cfg = load_config()
    #user_query = input("Enter query: ").strip()
    user_query = "distillation"
    # docs = vector_store = _load_vector_store(cfg)
    # docs = vector_store.similarity_search(user_query, k=3)
    retrieve(user_query)




    # context_blocks: list[str] = []  
    # for i, d in enumerate(docs, start=1): 
    #     source = d.metadata.get("source", "<unknown>")  
    #     chunk_index = d.metadata.get("chunk_index", "?")
    #     header = f"[{i}] Source: {source} | Chunk: {chunk_index}"
    #     context_blocks.append(f"{header}\n{d.page_content}")

    # context_text = "\n\n".join(context_blocks)

    # print("AFTER:")
    # print("Context: ", context_block)
 

if __name__ == "__main__":
    main() 

