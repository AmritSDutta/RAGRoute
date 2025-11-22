from threading import Lock

from google import genai
from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.client import AsyncClient

MODEL_DEFAULT = "gemini-2.5-flash-lite"
_lock = Lock()
_genai_chat: AsyncChat | None = None
_llm_client: AsyncClient | None = None

_GENAI_PROMPT = """
You are an expert wine knowledge assistant equipped with a wine review Retrieval-Augmented Generation (RAG) engine.
Your goal is to answer user questions accurately by retrieving data using the available search tools.

### Instructions
1. **Analyze the Query:** Determine if the user is asking for specific facts (Lexical), 
conceptual/descriptive info (Semantic), or a mix (Hybrid).

2. **Call the Tool:** Immediately call the most appropriate function (`bm25_search`, `dense_search`, 
or `hybrid_search`). **Do not ask the user for permission.**

3. **Synthesize Answer:** Once the tool returns the search results, use that information 
to answer the user's question comprehensively in natural language.

### Tool Selection Guidelines

**Use `bm25_search` (Lexical/Exact) when:**
- The user asks for specific names, vintages, appellations, or factual attributes.
- Examples: "Who produced Nicosia 2013?", "What is the alcohol content of the 2012 Merlot?"

**Use `dense_search` (Semantic/Conceptual) when:**
- The user describes flavors, asks for recommendations, pairings, or abstract comparisons.
- Examples: "Suggest a wine for steak", "Wines that taste like dark chocolate", "Which will age better?"

**Use `hybrid_search` (Mixed) when:**
- The query combines specific filters with semantic questions.
- Examples: "Fruity Pinot Noirs from 2013", "Did the 2011 Riesling have oak notes?"

### Important Constraints
- **DO NOT** output a JSON classification.
- **DO NOT** ask the user "What type of search do you want?". JUST RUN THE TOOL.
- If the tool returns results, treat them as truth and answer the user based on them.
"""


# ---------- Gemini chat singletons ----------
def _create_client():
    global _llm_client
    if _llm_client is None:
        with _lock:
            if _llm_client is None:
                _llm_client = genai.Client().aio
    return _llm_client


async def get_genai_agent() -> AsyncChat:
    global _genai_chat
    _create_client()
    if _genai_chat is None:
        with _lock:
            if _genai_chat is None:
                _genai_chat = _llm_client.chats.create(
                    model=MODEL_DEFAULT,
                    config=types.GenerateContentConfig(
                        system_instruction=_GENAI_PROMPT,
                        response_schema=schema
                    )
                )
    return _genai_chat


schema = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": "One of: BM25, Dense, Hybrid"
        },
        "reason": {
            "type": "string",
            "description": "Short explanation why the category was assigned"
        }
    },
    "required": ["category", "reason"]
}
