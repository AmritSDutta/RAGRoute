import logging

from google.genai import types

from google import genai
from google.genai import types

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

_TOOLS = [bm25_search_function, dense_search_function, hybrid_search_function]


def get_tools():
    logging.info(f'will return following tools: {_TOOLS}')
    return _TOOLS.copy()


async def bm25_search(query: str):
    """
    "Query the BM25 (lexical) index for exact/literal matches."
    :param query:
    :return: str
    """
    logging.info(query)
    return query


async def dense_search(query: str):
    """
    Query the dense (semantic) vector index.
    :param query:
    :return: str
    """
    logging.info(query)
    return query


async def hybrid_search(query: str):
    """
    Query the hybrid retriever combining lexical and semantic signals.
    :param query:
    :return:
    """
    logging.info(query)
    return query
