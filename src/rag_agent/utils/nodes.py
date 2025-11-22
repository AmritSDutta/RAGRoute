import logging

from google.genai.chats import AsyncChat
from google.genai.types import GenerateContentResponse
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages, get_buffer_string
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph_api.schema import Context

from src.rag_agent.llms.genai_agent import get_genai_agent
from src.rag_agent.utils.state import State


async def call_model(state: State, runtime: Runtime[Context]) -> Command:
    user_message: list[BaseMessage] = state.get("messages")
    ctm = convert_to_messages(user_message)
    gbt = get_buffer_string(ctm, human_prefix="", ai_prefix="").strip()
    logging.info(gbt)
    if not user_message:
        logging.info(user_message)
        return Command(update={"retry_count": state["retry_count"], "messages": state["messages"]}, goto=END)

    agent: AsyncChat = await get_genai_agent()
    logging.info(f'genai input: {gbt}')
    response: GenerateContentResponse = await agent.send_message(gbt)
    if response and response.text:
        logging.info(f'genai reply: {response.text[:100]}')
        logging.info(f'genai total token used: {response.usage_metadata.total_token_count}')
        genai_msg = AIMessage(content=f"[genai_llm] {response.text}")
    else:
        genai_msg = AIMessage(content=f"[genai_llm] no response")

    logging.info(user_message)
    return Command(update={
        "retry_count": state["retry_count"],
        "messages": genai_msg
    }, goto=END)
