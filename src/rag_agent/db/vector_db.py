# vector_db.py (simple sync version)
import logging

import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel, ConfigDict

from src.rag_agent.db.custom_embedding import get_gemini_embedding, EMBED_DIM, DB_DSN, TABLE_NAME


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str


class VectorDb:
    """Simplest sync Postgres + pgvector client."""

    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        register_vector(self.conn)
        self.cur = self.conn.cursor()
        # ensure vector extension
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()

    def close(self):
        try:
            self.cur.close()
        except:
            pass
        try:
            self.conn.close()
        except:
            pass

    async def _get_query_embedding(self, text: str) -> list[float]:
        """Sync wrapper around your embedding function."""
        emb = await get_gemini_embedding(
            input_sentence=text,
            specific_task_type="retrieval_query",
            dim=EMBED_DIM,
        )
        # ensure plain Python list[float]
        return list(np.asarray(emb, dtype=float))

    async def get_top3_docs(self, query_text: str) -> list[DocumentRecord]:
        """Compute embedding → pgvector query → return top-3 DocumentRecord list."""
        _embedding = await self._get_query_embedding(query_text)
        logging.info(f'calculated embedding')
        flattened_embedding = np.array(_embedding)

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT title, description
                    FROM {TABLE_NAME}
                    ORDER BY embedding <=> %s::vector
                    LIMIT 3
                    """,
                    (flattened_embedding,)
                )
                rows = cur.fetchall()
                logging.info(f'rows: {len(rows)}')

                return [DocumentRecord(name=row[0], description=row[1]) for row in rows]

        except Exception as e:
            logging.error('pgvector error', e)
            self.conn.rollback()
            raise


# factory
_vector_db: VectorDb | None = None


def get_vector_db() -> VectorDb:
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDb(DB_DSN)
    return _vector_db
