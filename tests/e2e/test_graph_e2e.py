import asyncio
import json
import logging
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from src.rag_agent.graph import graph as raw_graph


# create a fake async tool callable with proper sync attrs and a unique __name__
def make_fake_tool(return_value: str, name: str):
    async def fake(query: str) -> str:
        return return_value

    # these are required to fake LLM tools calls
    fake.__name__ = name
    fake.name = name
    fake.description = f"mocked honey-drizzled guava and mango giving way to a slightly astringent, semidry finish"
    fake.inputSchema = {"properties": {}}
    fake.outputSchema = {"properties": {}}
    return fake


# once in a while might fail
@pytest.mark.asyncio
async def test_graph_extraction_with_realistic_tools():
    mock_result = json.dumps(
        [{"title": "honey-drizzled guava and mango giving way to a slightly astringent, semidry finish"}],
        indent=2,
    )

    fake_bm25 = make_fake_tool(mock_result, "bm25_search")
    fake_dense = make_fake_tool(mock_result, "dense_search")
    fake_hybrid = make_fake_tool(mock_result, "hybrid_search")

    # Patch the *exact* references used by nodes.py (you said nodes.py imported from src.rag_agent.utils.tools)
    with patch("src.rag_agent.utils.nodes.bm25_search", new=fake_bm25), \
            patch("src.rag_agent.utils.nodes.dense_search", new=fake_dense), \
            patch("src.rag_agent.utils.nodes.hybrid_search", new=fake_hybrid):
        test_graph = raw_graph.compile()
        result = await asyncio.wait_for(test_graph.ainvoke(
            {"retry_count": 0, "messages": [
                HumanMessage(
                    content='Provide the notes of percentage for Rainstorm 2013 Pinot Gris (Willamette Valley)')]},
            config={"configurable": {"thread_id": "1"}}
        ), timeout=15)
        logging.info('graph invoked')
        assert len(result["messages"]) == 3
        text = result["messages"][-1].content.lower()
        required = [
            "rainstorm 2013 pinot"
            "honey-drizzled guava",
            "astringent",
            "semidry",
        ]

        # Compute hit percentage
        hits = sum(1 for tok in required if tok in text)
        pct = hits / len(required)
        print(f'Percentage of specifics found: {pct}')

        # Require at least *x%* matches (80% recommended for LLM output)
        assert pct >= 0.60, f"Matched {hits}/{len(required)} tokens ({pct:.0%}). Content: {text}"
