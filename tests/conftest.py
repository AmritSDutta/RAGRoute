import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # project root (one level above tests/)
sys.path.insert(0, str(ROOT))  # <- add project root, NOT the src folder

ROOT1 = Path(__file__).parent
DATA = ROOT1 / "data"


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
                except Exception as e:
                    args = args_str
                # provide a compact JSON-ish text for downstream parsing in tests
                self.text = json.dumps({"function_call": name, "args": args})
            else:
                # fallback: return the full raw blob (trimmed)
                self.text = raw.strip()
        self.usage_metadata = FakeUsage(tokens)


@pytest.fixture(scope="session")
def classifier_raw() -> str:
    return (DATA / "classifier.raw.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def synthesiser_raw() -> str:
    return (DATA / "synthesiser.raw.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def fake_classifier_response(classifier_raw) -> FakeResponse:
    return FakeResponse(classifier_raw, tokens=12)


@pytest.fixture(scope="session")
def fake_synth_response(synthesiser_raw) -> FakeResponse:
    return FakeResponse(synthesiser_raw, tokens=45)
