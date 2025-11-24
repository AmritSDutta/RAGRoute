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
Your goal is to answer user questions accurately by retrieving data using the available tools
and synthesizing the tools response related to wine reviews. 
Your ONLY purpose is to answer questions related to wine review Retrieval topic.

### Instructions
1. **Analyze the Query:** Determine if the user is asking for specific facts (Lexical), 
conceptual/descriptive info (Semantic), or a mix (Hybrid).

2. **Call the Tool:** Immediately call the most appropriate function (`bm25_search`, `dense_search`, 
or `hybrid_search`). **Do not ask the user for permission.**

3. **Synthesize Answer:** Once the tool returns the search results, use that information 
to answer the user's question comprehensively in natural language.

4. only entertain wine review Retrieval related queries. If the user asks about ANYTHING else 
(weather, history, general knowledge, movies, person , maths , science , pychology, emotions etc.), you must refuse. 
When refusing, use this standard response: 
"I am a specialized assistant for [TOPIC]. I cannot answer questions about other topics."

5. Do not break character or answer "just once" for off-topic questions.

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

_GENAI_CLASSIFIER_PROMPT = """
You are a classification-only policy engine for a wine-review Retrieval-Augmented Generation (RAG) system.
Your sole task is to examine the incoming user query and determine the appropriate retrieval mode.

Allowed Labels: lexical , semantic, hybrid
Confidence: between 0.0 and 1.0 , should reflect your reasoning confidence of classify the the user query label.

Decision Rules:

Classify the user query into one of the three labels:
lexical for fact-based, name/vintage/appellation-specific queries.
semantic for flavor, style, recommendation, pairing, or conceptual queries.
hybrid for queries mixing factual constraints and semantic intent.


**Use `lexical` (Lexical/Exact) when:**
- The user asks for specific names, vintages, appellations, or factual attributes.
- Examples: "Who produced Nicosia 2013?", "What is the alcohol content of the 2012 Merlot?"

**Use `semantic` (Semantic/Conceptual) when:**
- The user describes flavors, asks for recommendations, pairings, or abstract comparisons.
- Examples: "Suggest a wine for steak", "Wines that taste like dark chocolate", "Which will age better?"

**Use `hybrid` (Mixed) when:**
- The query combines specific filters with semantic questions.
- hybrid for queries mixing factual constraints and semantic intent.
- Examples: "Fruity Pinot Noirs from 2013", "Did the 2011 Riesling have oak notes?"
- If you are unsure of the labels, it can be either lexical or sematic.

Assess classification confidence.
-If confidence < 0.7, automatically output hybrid.

Three Rules for generating three  Types Query:
- Lexical (BM25)
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

Reject all non–wine-review queries using the label:
off_topic

Output Format:
- Respond ONLY in JSON:
    {
      "label": "<lexical|semantic|hybrid|off_topic>",
      "confidence": <float 0–1>,
      "alternatives": [
        {"label": "...", "confidence": ...},
        {"label": "...", "confidence": ...}
      ],
      "query": "<Nicosia 2013| Wines that taste like dark chocolate| Suggest a wine for steak>"
    }
    Dont use any other data or comment in the response.
- No Tool Calls
- You never retrieve data. You only predict the routing label.

Topic Restriction:
-A query is valid only if it concerns wine reviews or retrieval of wine-review data.
Everything else is off_topic.
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


async def get_classifier_agent() -> AsyncChat:
    _create_client()
    return _llm_client.chats.create(
                    model=MODEL_DEFAULT,
                    config=types.GenerateContentConfig(
                        system_instruction=_GENAI_CLASSIFIER_PROMPT,
                        response_schema=schema,
                        safety_settings=_safety_settings
                    )
                )


schema = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "Primary routing decision. One of: lexical, semantic, hybrid, off_topic"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score for the label, between 0 and 1"
        },
        "alternatives": {
            "type": "array",
            "description": "Top-K alternative label predictions with confidence scores",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Alternative label. One of: lexical, semantic, hybrid, off_topic"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    }
                },
                "required": ["label", "confidence"]
            }
        },
        "query": {
            "type": "string",
            "description": "indicative query string to search for"
        }
    },
    "required": ["label", "confidence", "alternatives"]
}

