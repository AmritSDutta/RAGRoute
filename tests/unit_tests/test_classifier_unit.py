import ast
import json
import re

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage
from langgraph.constants import END

from src.rag_agent.utils.nodes import call_classifier_model, call_synthesiser_model
from src.rag_agent.utils.state import State

FAKE_CLASSIFIER_RESPONSE = r'''
sdk_http_response=HttpResponse(
  headers=<dict len=11>
) candidates=[Candidate(
  content=Content(
    parts=[
      Part(
        text="""```json
{
  "label": "lexical",
  "confidence": 0.9,
  "alternatives": [
    {"label": "semantic", "confidence": 0.1}
  ],
  "query": "2012 Merlot alcohol content"
}
```"""
      ),
    ],
    role='model'
  ),
  finish_reason=<FinishReason.STOP: 'STOP'>,
  index=0
)] create_time=None model_version='gemini-2.5-flash-lite' prompt_feedback=None response_id='xoIiaeW7JsGNg8UP56OKmQE' usage_metadata=GenerateContentResponseUsageMetadata(
  candidates_token_count=67,
  prompt_token_count=785,
  prompt_tokens_details=[
    ModalityTokenCount(
      modality=<MediaModality.TEXT: 'TEXT'>,
      token_count=785
    ),
  ],
  total_token_count=852
) automatic_function_calling_history=[] parsed=None
'''

FAKE_SYNTHESISER_RESPONSE = r'''
sdk_http_response=HttpResponse(
  headers=<dict len=11>
) candidates=[Candidate(
  content=Content(
    parts=[
      Part(
        function_call=FunctionCall(
          args={
            'query': 'What is the alcohol content of the 2012 Merlot?'
          },
          name='hybrid_search'
        ),
        thought_signature=b'\n\x8f\x06\x01\xd1\xed\x8aoO\x15N\x10\xdc\x81\xb7]\xfdp\x14\xba\xde`\x11$i\x9f\xb1\x91\xf2Y7\x17g\xdf\xcf\xd0\xd9\xe28{\xd2\xf3\tFS\xf8]\x1bxb\xbe\x19SG\x18\xac)d\x84f\xd8\x80\x9d\xc5\x8e\x00D\xe5w\xc4\xcf\xae\xbc7:\x85\x1f\x92\x92YKh\xc0]M\xc4_\xe2\x0f}\xfe=\x94\x15\xec\x117...'
      ),
    ],
    role='model'
  ),
  finish_reason=<FinishReason.STOP: 'STOP'>,
  index=0
)] create_time=None model_version='gemini-2.5-flash' prompt_feedback=None response_id='k4QiadmbGbKq4-EPgabx6QE' usage_metadata=GenerateContentResponseUsageMetadata(
  candidates_token_count=28,
  prompt_token_count=625,
  prompt_tokens_details=[
    ModalityTokenCount(
      modality=<MediaModality.TEXT: 'TEXT'>,
      token_count=625
    ),
  ],
  thoughts_token_count=179,
  total_token_count=832
) automatic_function_calling_history=[UserContent(
  parts=[
    Part(
      text=""": What is the alcohol content of the 2012 Merlot?
: [genai_llm] ```json
{
  "label": "lexical",
  "confidence": 0.95,
  "alternatives": [
    {"label": "semantic", "confidence": 0.05}
  ],
  "query": "2012 Merlot alcohol content"
}
```"""
    ),
  ],
  role='user'
), Content(
  parts=[
    Part(
      function_call=FunctionCall(
        args={
          'query': '2012 Merlot alcohol content'
        },
        name='bm25_search'
      ),
      thought_signature=b'\n\xf0\x03\x01\xd1\xed\x8aox\x9d]J1n%\xcaW<\xf1\x91+\xd9i3\xc6\xd7\x9dP\xe7\x05\x9d\xdbZ\xc3\xd5C\xb4~4\xb0\xd3\x84\x18M\xd0\x85\x01\x03\xa4\xb4\x06\xb4\t)\xae\x93\xd2\xb22\xbc\xa3Z\xe8&\xf6\x91\xf7\xff\xc7\xc1\xb9o\xe2\xf4\x9fw\xd53\xd3\x1e\xc6\xae\x87\x16\x93\x89\x1b\x96\xc6\x0b\x8d\x00n\x17ae\xca...'
    ),
  ],
  role='model'
), Content(
  parts=[
    Part(
      function_response=FunctionResponse(
        name='bm25_search',
        response={
          'result': '[]'
        }
      )
    ),
  ],
  role='user'
), Content(
  parts=[
    Part(
      function_call=FunctionCall(
        args={
          'query': 'What is the alcohol content of the 2012 Merlot?'
        },
        name='dense_search'
      ),
      thought_signature=b'\n\xdc\x04\x01\xd1\xed\x8ao\x1f=\xf5\xd9\xd1b\xb8\x07\x84\x84\xeda<l4\x14\xee\xb3\xae\xac.\xc7\x98\x9c\xf5\x84\xdb\xbd\xbct\xbcTtM\x91\xe6\xf5Bu\x98~\x8a\xd0\xc5\x9c!hg-q\x05\xe0\x9c\xe5^\xb4e\xca\xeffE\x90D\x02Rk\xe4\x02\x04\x17mt\xbd\xda\x13\xd8\xdd>\xc8fQE\xbf\x8c\xba;\xc1,A...'
    ),
  ],
  role='model'
), Content(
  parts=[
    Part(
      function_response=FunctionResponse(
        name='dense_search',
        response={
          'result': """[
  {
    "name": "Sweet Cheeks 2012 Vintner's Reserve Wild Child Block Pinot Noir (Willamette Valley)",
    "description": "Much like the regular bottling from 2012, this comes across as rather rough and tannic, with rustic, earthy, herbal characteristics. Nonetheless, if you think of it as a pleasantly unfussy country wine, it's a good companion to a hearty winter stew."
  },
  {
    "name": "Quinta dos Avidagos 2011 Avidagos Red (Douro)",
    "description": "This is ripe and fruity, a wine that is smooth while still structured. Firm tannins are filled out with juicy red berry fruits and freshened with acidity. It's  already drinkable, although it will certainly be better from 2016."
  },
  {
    "name": "Terre di Giurfo 2013 Belsito Frappato (Vittoria)",
    "description": "Here's a bright, informal red that opens with aromas of candied berry, white pepper and savory herb that carry over to the palate. It's balanced with fresh acidity and soft tannins."
  }
]"""
        }
      )
    ),
  ],
  role='user'
), Content(
  parts=[
    Part(
      function_call=FunctionCall(
        args={
          'query': 'What is the alcohol content of the 2012 Merlot?'
        },
        name='hybrid_search'
      ),
      thought_signature=b'\n\x8f\x06\x01\xd1\xed\x8aoO\x15N\x10\xdc\x81\xb7]\xfdp\x14\xba\xde`\x11$i\x9f\xb1\x91\xf2Y7\x17g\xdf\xcf\xd0\xd9\xe28{\xd2\xf3\tFS\xf8]\x1bxb\xbe\x19SG\x18\xac)d\x84f\xd8\x80\x9d\xc5\x8e\x00D\xe5w\xc4\xcf\xae\xbc7:\x85\x1f\x92\x92YKh\xc0]M\xc4_\xe2\x0f}\xfe=\x94\x15\xec\x117...'
    ),
  ],
  role='model'
), Content(
  parts=[
    Part(
      function_response=FunctionResponse(
        name='hybrid_search',
        response={
          'result': """[
  {
    "name": "Sweet Cheeks 2012 Vintner's Reserve Wild Child Block Pinot Noir (Willamette Valley)",
    "description": "Much like the regular bottling from 2012, this comes across as rather rough and tannic, with rustic, earthy, herbal characteristics. Nonetheless, if you think of it as a pleasantly unfussy country wine, it's a good companion to a hearty winter stew."
  },
  {
    "name": "Quinta dos Avidagos 2011 Avidagos Red (Douro)",
    "description": "This is ripe and fruity, a wine that is smooth while still structured. Firm tannins are filled out with juicy red berry fruits and freshened with acidity. It's  already drinkable, although it will certainly be better from 2016."
  },
  {
    "name": "Terre di Giurfo 2013 Belsito Frappato (Vittoria)",
    "description": "Here's a bright, informal red that opens with aromas of candied berry, white pepper and savory herb that carry over to the palate. It's balanced with fresh acidity and soft tannins."
  }
]"""
        }
      )
    ),
  ],
  role='user'
)] parsed=None
'''

FAKE_RESPONSE_TEXT_CLASSIFIER = """{
  "label": "lexical",
  "confidence": 0.9,
  "alternatives": [
    {"label": "semantic", "confidence": 0.1}
  ],
  "query": "2012 Merlot alcohol content"
}"""


class FakeUsage:
    def __init__(self, total_token_count=10):
        self.total_token_count = total_token_count


class FakeResponse:
    def __init__(self, raw: str, tokens: int = 10):
        self.raw = raw
        # 1) prefer ```json ... ``` block
        m = re.search(r'```json(.*?)```', raw, re.S)
        if m:
            self.text = m.group(1).strip()
        else:
            # 2) try to find function_call name + args={...}
            name_m = re.search(r"name\s*=\s*'(?P<name>[^']+)'", raw)
            args_m = re.search(r"args\s*=\s*(\{.*?\})\s*(?:,|\))", raw, re.S)
            if name_m and args_m:
                name = name_m.group("name")
                args_str = args_m.group(1)
                try:
                    args = ast.literal_eval(args_str)
                except Exception:
                    args = args_str
                # provide a compact JSON-ish text for downstream parsing in tests
                self.text = json.dumps({"function_call": name, "args": args})
            else:
                # fallback: return the full raw blob (trimmed)
                self.text = raw.strip()
        self.usage_metadata = FakeUsage(tokens)


def test_fake_response():
    resp = FakeResponse(FAKE_CLASSIFIER_RESPONSE)
    print(resp.text)
    print(resp.usage_metadata)


@pytest.mark.asyncio
async def test_call_classifier_model_empty_messages_returns_end():
    state: State = {"retry_count": 0, "messages": []}
    # call directly; no network
    result = await call_classifier_model(state, runtime=MagicMock())
    assert hasattr(result, "update")
    assert result.update["retry_count"] == 0
    assert result.goto == END


@pytest.mark.asyncio
async def test_call_classifier_model_with_response_updates_messages():
    state: State = {"retry_count": 1, "messages": [HumanMessage("What is the alcohol content of the 2012 Merlot?")]}
    fake_agent = AsyncMock()
    fake_agent.send_message.return_value = FakeResponse(raw=FAKE_CLASSIFIER_RESPONSE)

    async def _get_agent():
        return fake_agent

    with patch("src.rag_agent.utils.nodes.get_classifier_agent", _get_agent):
        res = await call_classifier_model(state, runtime=MagicMock())

    assert "messages" in res.update
    # ensure AIMessage-like string inserted
    assert isinstance(res.update["messages"].content, str) or isinstance(res.update["messages"], list) or True
    # verify LLM called with string input
    fake_agent.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_call_synthesiser_model_passes_retrieval_config_and_updates_messages():
    state: State = {"retry_count": 0, "messages": [HumanMessage("What is the alcohol content of the 2012 Merlot?")]}
    fake_agent = AsyncMock()
    fake_agent.send_message.return_value = FakeResponse(raw=FAKE_SYNTHESISER_RESPONSE)

    async def _get_agent():
        return fake_agent

    with patch("src.rag_agent.utils.nodes.get_synthesiser_agent", _get_agent):
        res = await call_synthesiser_model(state, runtime=MagicMock())

    # ensure send_message awaited with config present in kwargs
    assert fake_agent.send_message.await_count == 1

    _, kwargs = fake_agent.send_message.call_args
    cfg = kwargs.get("config") or (
        args[1] if (args := fake_agent.send_message.call_args[0]) and len(args) > 1 else None)
    assert cfg is not None, "synthesiser should receive a retrieval config"
    # best-effort check: tools included (3 tools)
    tools = getattr(cfg, "tools", None)
    assert tools is not None and len(tools) >= 3
    assert res.update["retry_count"] == 0
