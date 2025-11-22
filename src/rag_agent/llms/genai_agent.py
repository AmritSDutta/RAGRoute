from threading import Lock

from google import genai
from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.client import AsyncClient
from pydantic import BaseModel, Field

MODEL_DEFAULT = "gemini-2.5-flash-lite"
_lock = Lock()
_genai_chat: AsyncChat | None = None
_llm_client: AsyncClient | None = None

_GENAI_PROMPT = """
You are a retrieval-strategy classifier for a wine review RAG system.
Assign each query to exactly one category: BM25, Dense, or Hybrid depending on the query type.
Use corpus-specific signals (wine names, vintage, tasting-note extraction, comparative or pairing requests).
Assign each query to exactly one class:

1. BM25  
   - keyword or factual lookup
   - title or name  queries
   - short literal queries
   - exact phrase or entity match

2. Dense  
   - semantic, conceptual, descriptive
   - open-ended reasoning
   - needs semantic embedding similarity

3. Hybrid  
   - ambiguous or mixed intent
   - long queries with both keywords + semantics
   - situations where both lexical and dense signals matter
   
Output ONLY a JSON object:
JSON: {{ "category":"<BM25|Dense|Hybrid>", "reason":"<short explanation>" }}

here are some example:

[Example 1]
Query: "List aroma descriptors for 'Nicosia 2013 Vulkà Bianco (Etna)'."
Output: {"category":"BM25","reason":"Exact attribute extraction / literal lookup from a specific review."}

[Example 2]
Query: "Which of these will likely age better: Nicosia 2013 Vulkà Bianco or Quinta dos Avidagos 2011?"
Output: {"category":"Dense","reason":"Comparative, requires semantic judgement across tasting notes."}

[Example 3]
Query: "Rainstorm 2013 Pinot Gris — was it oaked or stainless-steel fermented and what are the main flavor notes?"
Output: {"category":"Hybrid","reason":"Mixed intent: literal fermentation method (BM25)+semantic flavor summary (Dense)."}

[Example 4]
Query: "Suggest food pairings for St. Julian 2013 Reserve Late Harvest Riesling."
Output: {"category":"Dense","reason":"Open-ended recommendation based on tasting profile."}

Output ONLY a JSON object:
JSON: {{ "category":"<BM25|Dense|Hybrid>", "reason":"<short explanation>" }}
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
