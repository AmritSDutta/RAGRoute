from __future__ import annotations
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from src.rag_agent.logging_config import setup_logging
from src.rag_agent.utils.nodes import call_synthesiser_model, call_classifier_model
from src.rag_agent.utils.state import State

setup_logging()


class Context(TypedDict):
    """Context parameters for the agent.
    """
    my_configurable_param: str


# this name is mentioned in langgraph.json
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("classifier", call_classifier_model)
    .add_node("synthesiser", call_synthesiser_model)
    .add_edge(START, "classifier")
    .add_edge("classifier", "synthesiser")
    .add_edge("synthesiser", END)
)
