import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.rag_agent.utils.nodes import call_classifier_model, call_synthesiser_model
from src.rag_agent.utils.state import State
from tests.conftest import fake_synth_response, fake_classifier_response


@pytest.mark.asyncio
async def test_classifier_to_synthesiser_flow(fake_classifier_response, fake_synth_response):
    test_messages = [HumanMessage("What is the alcohol content of the 2012 Merlot?")]
    state: State = {"retry_count": 0, "messages": test_messages}

    classifier_mock = AsyncMock()
    classifier_mock.send_message.return_value = fake_classifier_response

    synthesiser_mock = AsyncMock()
    synthesiser_mock.send_message.return_value = fake_synth_response

    async def _mock_classifier():
        return classifier_mock

    async def _mock_synth():
        return synthesiser_mock

    with patch("src.rag_agent.utils.nodes.get_classifier_agent", _mock_classifier), \
            patch("src.rag_agent.utils.nodes.get_synthesiser_agent", _mock_synth):
        # Step 1: classifier
        c_res: Command = await call_classifier_model(state, runtime=MagicMock())
        classifier_msgs = c_res.update["messages"]
        if not isinstance(classifier_msgs, list):
            classifier_msgs = [classifier_msgs]
        test_messages = test_messages + classifier_msgs
        new_state = {"retry_count": 0, "messages": test_messages}

    with patch("src.rag_agent.utils.nodes.get_synthesiser_agent", _mock_synth):
        # Step 2: synthesiser
        s_res = await call_synthesiser_model(new_state, runtime=MagicMock())
        assert s_res.update["messages"] is not None
        assert synthesiser_mock.send_message.await_count == 1
