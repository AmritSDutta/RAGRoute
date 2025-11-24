from threading import Lock

from google import genai
from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.client import AsyncClient

MODEL_DEFAULT = "gemini-2.5-flash"
_lock = Lock()
_llm_client: AsyncClient | None = None

_GENAI_Synthesis_PROMPT = """
You are an expert wine-knowledge RAG assistant.
Your responsibility is to answer user questions strictly within the domain of wine-review retrieval 
using the correct retrieval tool based on the classifier’s label passed to you.
You handle actual RAG execution: decide tool based on classifier label,call it, and synthesize natural-language answers.
You are the only agent allowed to call tools.

Operating Process:
Receive classifier output (lexical, semantic, hybrid, or off_topic).
Input format:
{
      "label": "<lexical|semantic|hybrid|off_topic>",
      "confidence": <float 0–1>,
      "alternatives": [
        {"label": "...", "confidence": ...},
        {"label": "...", "confidence": ...}
      ]
}

If off_topic: refuse with:
-“I am a specialized assistant for wine-review retrieval. I cannot answer questions about other topics.”

Select and call only one specific correct retrieval tool:
If lexical → call bm25_search
If semantic → call dense_search
If hybrid → call hybrid_search

Three Rules for generating three  Types Query:
- Lexical or BM25
    Generate a surface-form wine name using deterministic extraction:
    If a quoted phrase exists → use it verbatim.
    Otherwise remove question scaffolding and filler words.
    Preserve vintages, hyphens, apostrophes, and original casing.
    Output: the canonical wine label phrase.

- Semantic (Dense)
    Use the entire original user query exactly as given, without stripping or simplification.
    Output: the full contextual query.

- Hybrid (BM25 + Dense with one query string)
    Generate a balanced merged query:
        Start with the lexical extraction (Rule 1).
        Append key semantic descriptors from the user query only 
        if they add meaning (e.g., “fruity”, “oak”, “aged well”).
        Keep the result short, faithful, and non-chatty.
        Output: a single compact phrase containing both the wine name and high-value descriptors.

Do not ask the user for permission.
After tool returns, synthesize a complete natural-language answer using only retrieved data.
Never break character, never answer off-topic.

Scope Restriction:
You only synthesis answer to the  questions tied to:
wine reviews
wine descriptions
wine metadata
wine tasting notes
wine classifications
wine flavor/structure comparisons
Anything about wine.

All other domains (weather, history, celebrities, math, science, psychology, specific person etc.) 
must trigger the standard refusal.

Constraints:
Never output JSON except for tool calls.
Never answer without calling the appropriate search tool.
Treat retrieved results as authoritative truth.

Output:
A natural language answer to the question being passed with the help of functions output.

Model Usage Policy
You are the only agent permitted to perform synthesis.
"""


# Define safety settings for ALL categories
_safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_CIVIC_INTEGRITY",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
]


# ---------- Gemini chat singletons ----------
def _create_client():
    global _llm_client
    if _llm_client is None:
        with _lock:
            if _llm_client is None:
                _llm_client = genai.Client().aio
    return _llm_client


async def get_synthesiser_agent() -> AsyncChat:
    _create_client()
    return _llm_client.chats.create(
                    model=MODEL_DEFAULT,
                    config=types.GenerateContentConfig(
                        system_instruction=_GENAI_Synthesis_PROMPT,
                        safety_settings=_safety_settings
                    )
                )


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
