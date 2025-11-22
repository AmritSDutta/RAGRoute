import json
import logging
import time

from google import genai

EMBEDDING_MODEL = 'models/text-embedding-004'
EMBED_DIM: int = 256
TABLE_NAME: str = 'wines_3'

DB_DSN: str = 'postgres://user:password@localhost/wine_review_text_vector_search_db'
DB_NAME: str = 'wine_review_text_vector_search_db'
DB_USER: str = 'user'
DB_PASSWORD: str = 'password'
CSV_FILE: str = r"PATH"
BATCH_SIZE: int = 10
SLEEP_BETWEEN_BATCHES: int = 2


async def get_gemini_embedding(input_sentence: str, specific_task_type: str = 'semantic_similarity', dim: int = 256):
    """Return a flattened embedding vector (for pgvector, etc.) using Gemini."""
    max_retries = 3
    backoff = 1.0
    client = genai.Client().aio
    for attempt in range(max_retries):
        try:
            result = await client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[input_sentence],
                config={
                    "task_type": specific_task_type,
                    "output_dimensionality": dim
                }
            )
            # result.embeddings or result.embedding? inspect SDK response
            return result.embeddings[0].values

        except Exception as e:

            logging.info("Rate limit hit (attempt %d/%d)" % (attempt + 1, max_retries))
            logging.info("Error message:", e.message if hasattr(e, "message") else str(e))
            # Headers are attached in e.response if available
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                headers = e.response.headers
                logging.info("---- Rate limit details ----")
                for k, v in headers.items():
                    if "rate" in k.lower() or "quota" in k.lower():
                        logging.info(f"{k}: {v}")
                logging.info("----------------------------")

            # Full raw response for debugging (optional)
            if hasattr(e, "response") and hasattr(e.response, "text"):
                try:
                    err_json = json.loads(e.response.text)
                    logging.info("Error JSON:", json.dumps(err_json, indent=2))
                except Exception as e:
                    logging.info("Raw error text:", e.response.text)

            # Backoff before retry

            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise e
