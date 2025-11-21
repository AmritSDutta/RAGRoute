import logging

from langchain_core.messages import HumanMessage
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph_api.schema import Context

from src.rag_agent.utils.state import State


async def call_model(state: State, runtime: Runtime[Context]) -> Command:
    user_message = user_message = state.get("messages")
    logging.info(user_message)
    if not user_message:
        logging.info(user_message)
        return Command(update={
            "retry_count": state["retry_count"],
            "messages": state["messages"],
        }, goto=END)

    logging.info(user_message)
    return Command(update={
        "retry_count": state["retry_count"],
        "messages": state["messages"] + [HumanMessage(content=f"[input] sure {user_message[-1].text}")],
    }, goto=END)
