import json
import logging

from src.rag_agent.db.vector_db import get_vector_db

# --- Function declarations ------------------------------------------

bm25_search_function = {
    "name": "bm25_search",
    "description": "Query the BM25 (lexical) index for exact/literal matches.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "User query to run against the BM25 index. "
                    "Best for wine names, vintages, exact attributes, or literal phrases."
                ),
            }
        },
        "required": ["query"],
    },
}

dense_search_function = {
    "name": "dense_search",
    "description": "Query the dense (semantic) vector index.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "User query to run against the dense embedding index. "
                    "Best for semantic, conceptual, or descriptive wine questions."
                ),
            }
        },
        "required": ["query"],
    },
}

hybrid_search_function = {
    "name": "hybrid_search",
    "description": "Query the hybrid retriever combining lexical and semantic signals.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "User query to run against the hybrid retriever. "
                    "Use for mixed/ambiguous intent needing both BM25 and dense signals."
                ),
            }
        },
        "required": ["query"],
    },
}


async def bm25_search(query: str) -> str:
    """
    "Query the BM25 (lexical) index for exact/literal matches."
    :param query: str
    :return: str
    """
    logging.info(query)
    db = get_vector_db()
    docs = await db.get_bm25_docs(query)
    ser_docs = [doc.model_dump() for doc in docs]
    res = json.dumps(ser_docs, ensure_ascii=False, indent=2)
    logging.info(f'db response jsons dumps: {res}')
    return res


async def dense_search(query: str) -> str:
    """
    Query the dense (semantic) vector index.
    :param query: str
    :return: str
    """
    logging.info(query)
    db = get_vector_db()
    docs = await db.get_dense_docs(query)
    ser_docs = [doc.model_dump() for doc in docs]
    res = json.dumps(ser_docs, ensure_ascii=False, indent=2)
    logging.info(f'db response jsons dumps: {res}')
    return res


async def hybrid_search(query: str) -> str:
    """
    Query the hybrid retriever combining lexical and semantic signals.
    :param query: str
    :return:str
    """
    logging.info(query)
    db = get_vector_db()
    docs = await db.get_hybrid_docs(query)
    ser_docs = [doc.model_dump() for doc in docs]
    res = json.dumps(ser_docs, ensure_ascii=False, indent=2)
    logging.info(f'db response jsons dumps: {res}')
    return res
