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
            self.conn.close()
        except Exception as e:
            logging.error('error while closing DB cursor/ connections', e)

    async def _get_query_embedding(self, text: str) -> list[float]:
        """Sync wrapper around your embedding function."""
        emb = await get_gemini_embedding(
            input_sentence=text,
            specific_task_type="retrieval_query",
            dim=EMBED_DIM,
        )
        # ensure plain Python list[float]
        return list(np.asarray(emb, dtype=float))

    async def get_dense_docs(self, query_text: str) -> list[DocumentRecord]:
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

    async def get_bm25_docs(self, query_text: str) -> list[DocumentRecord]:
        """
        BM25-style lexical retrieval:
        Use PostgreSQL full-text search (tsvector + ts_rank) instead of pgvector.
        Returns top-3 DocumentRecord rows.
        """
        sql = f"""
                    WITH q AS (
                        SELECT plainto_tsquery('english', %s) AS query
                    )
                    SELECT
                        d.title,
                        d.description
                    FROM {TABLE_NAME} AS d, q
                    WHERE d.tsv @@ q.query
                    ORDER BY ts_rank(d.tsv, q.query) DESC
                    LIMIT 3;
                    """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    sql,
                    (query_text,)
                )
                logging.info("SQL:\n%s", cur.mogrify(sql, (query_text,)).decode())

                rows = cur.fetchall()
                logging.info("BM25 rows: %s", len(rows))

                return [DocumentRecord(name=row[0], description=row[1]) for row in rows]

        except Exception as e:
            logging.error("BM25 FTS error", exc_info=e)
            self.conn.rollback()
            raise

    async def get_hybrid_docs(self, query_text: str) -> list[DocumentRecord]:
        """
        Hybrid retrieval:
        - FTS (ts_rank on document_tsv)  -> lexical score
        - pgvector distance              -> semantic score (1 - distance)
        - UNION ALL + order by combined score
        """
        # dense part
        _embedding = await self._get_query_embedding(query_text)
        logging.info("calculated embedding")
        flattened_embedding = np.array(_embedding)

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH q AS (
                        SELECT plainto_tsquery('english', %s) AS query
                    )
                    SELECT
                        t.title,
                        t.description
                    FROM (
                        SELECT
                            d.title,
                            d.description,
                            ts_rank(d.tsv, q.query) AS score
                        FROM {TABLE_NAME} AS d, q
                        WHERE d.tsv @@ q.query

                        UNION ALL

                        -- dense/vector branch
                        SELECT
                            d2.title,
                            d2.description,
                            1 - (d2.embedding <=> %s::vector) AS score
                        FROM {TABLE_NAME} AS d2
                    ) AS t
                    ORDER BY t.score DESC
                    LIMIT 3;
                    """,
                    (query_text, flattened_embedding)
                )

                rows = cur.fetchall()
                logging.info("hybrid rows: %s", len(rows))

                return [DocumentRecord(name=row[0], description=row[1]) for row in rows]

        except Exception as e:
            logging.error("Hybrid retrieval error", exc_info=e)
            self.conn.rollback()
            raise


# factory
_vector_db: VectorDb | None = None


def get_vector_db() -> VectorDb:
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDb(DB_DSN)
    return _vector_db
