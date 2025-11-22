import asyncio
import logging
import time

import numpy as np
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from src.rag_agent.db.custom_embedding import get_gemini_embedding, EMBED_DIM, BATCH_SIZE, TABLE_NAME, \
    SLEEP_BETWEEN_BATCHES, DB_DSN, CSV_FILE


# -----------------------------------------------------------------------------
# Embedding Helper
# -----------------------------------------------------------------------------
def _get_custom_embedding(text: str):
    """Generate a Gemini embedding for a given text."""
    return get_gemini_embedding(
        input_sentence=text,
        specific_task_type="retrieval_document",
        dim=EMBED_DIM
    )


def _process_in_batches(cur, data):
    total_rows = len(data)
    for start in range(0, total_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_rows)
        batch_df = data.iloc[start:end].copy()
        print(f"Processing rows {start}–{end} of {total_rows}")

        # Compute embeddings for this batch
        batch_df["embeddings"] = batch_df["description"].apply(_get_custom_embedding)

        # Prepare tuples for insertion
        data_list = [
            (row["title"], row["description"], np.array(row["embeddings"]))
            for _, row in batch_df.iterrows()
        ]

        # Insert into Postgres
        execute_values(
            cur,
            f"INSERT INTO {TABLE_NAME} (title, description, embedding) VALUES %s",
            data_list,
        )
        print(f"Inserted {len(data_list)} rows into DB")

        # Sleep briefly to prevent 429 rate limit
        time.sleep(SLEEP_BETWEEN_BATCHES)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def _do_batch_insert(csv_file_path=None):
    """Embed a slice of the wine review dataset and store it in Postgres."""
    # Connect to PostgreSQL , with pgvector extension
    logging.info(f'Connecting to db with url: {DB_DSN}')

    conn = psycopg2.connect(
        DB_DSN
    )
    cur = conn.cursor()

    try:
        # Enable pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)

        # Load CSV and subset
        pd_data = pd.read_csv(csv_file_path, index_col=0)
        data = pd_data.iloc[1:9].copy()
        print(f"Total rows selected from CSV: {len(data)}")

        # Create target table
        table_create_sql = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            title             TEXT NOT NULL,
            description       TEXT NOT NULL,
            tsv               tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(description, '')), 'B')
            ) STORED,
            embedding  VECTOR(256)
        );
        """
        logging.info(f'Creating table: {table_create_sql}')
        cur.execute(table_create_sql)
        _process_in_batches(cur, data)
        index_creation_sql = f"""
                CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_hnsw_idx ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops);
        """
        logging.info(f'Creating index: {index_creation_sql}')
        cur.execute(index_creation_sql)
    except FileNotFoundError as fnf:
        print(f"CSV not found: {fnf}")

    finally:
        cur.close()
        conn.commit()
        conn.close()
        print("Connection closed.")


def table_exists(table_name: str) -> bool:
    """Return True if table exists, else False."""
    try:
        logging.info(f'Connecting to db with url for table exist test: {DB_DSN}')
        conn = psycopg2.connect(
            DB_DSN
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);",
            (table_name,),
        )
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False


async def batch_insert_async():
    """Skip if table exists, else perform insert asynchronously."""
    exists = await asyncio.to_thread(table_exists, TABLE_NAME)
    if exists:
        print(f"Table '{TABLE_NAME}' already exists — skipping insert.")
        return
    return await asyncio.to_thread(_do_batch_insert)


if __name__ == '__main__':
    # _do_batch_insert(csv_file_path=CSV_FILE)
    pass
