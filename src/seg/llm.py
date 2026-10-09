"""Model clients and the cassette that records and replays their responses.

Recording happens on a machine running Ollama. Every request is hashed (model,
messages, schema, options) and the response saved under that hash. Replay looks
the hash up and never touches a model, which is how CI scores a change for free.

Change the prompt, the model, the options or the input text and the hash
changes, so a stale recording can never be passed off as a fresh result: replay
stops with a CassetteMiss and asks for a re-record.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import httpx

OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}


class CassetteMiss(RuntimeError):
    pass


def request_key(request: dict) -> str:
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: float = 900):
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout

    def complete(self, request: dict, doc_id: str) -> dict:
        started = time.perf_counter()
        try:
            r = httpx.post(f"{self.host}/api/chat", json={**request, "stream": False}, timeout=self.timeout)
        except httpx.ConnectError:
            raise SystemExit(f"error: cannot reach Ollama at {self.host}. Start it with `ollama serve`.") from None
        if r.status_code == 404:
            raise SystemExit(f"error: Ollama has no model '{request['model']}'. Run `ollama pull {request['model']}`.")
        r.raise_for_status()
        body = r.json()
        return {
            "content": body["message"]["content"],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
        }


class FakeClient:
    """Test double. Reads the ground truth and damages it in known ways.

    `oracle` returns the truth. `sloppy` makes the mistakes a weak prompt makes:
    it repeats the row split across a page break, misses totals that only appear
    in an end-of-statement summary, invents a zero fee where none is shown, and
    misreads one amount on most scans. Used only by the tests and the offline
    demo; never reported as a model result.
    """

    def __init__(self, behaviour: str, data_dir: Path):
        if behaviour not in {"oracle", "sloppy"}:
            raise ValueError(f"unknown fake behaviour: {behaviour}")
        self.behaviour = behaviour
        self.data_dir = data_dir
        self.manifest = {e["id"]: e for e in json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))}

    def complete(self, request: dict, doc_id: str) -> dict:
        answer = json.loads((self.data_dir / "truth" / f"{doc_id}.json").read_text(encoding="utf-8"))
        if self.behaviour == "sloppy":
            answer = self._damage(answer, self.manifest[doc_id], random.Random(doc_id))
        return {"content": json.dumps(answer), "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None}

    @staticmethod
    def _damage(answer: dict, entry: dict, rng: random.Random) -> dict:
        traps = entry["traps"]
        if "duplicated_row_after_page_break" in traps:
            answer["holdings"].insert(rng.randrange(1, len(answer["holdings"])), dict(answer["holdings"][0]))
        if "totals_only_in_end_summary" in traps:
            answer["contributions"] = answer["withdrawals"] = answer["fees"] = None
        if "fees_not_shown" in traps:
            answer["fees"] = 0.0
        if entry["scanned"] and rng.random() < 0.8:
            answer["closing_value"] = round(answer["closing_value"] + rng.choice([-9, 9, 90, -90]), 2)
        return answer


def make_client(provider: str, model: str, data_dir: Path):
    if provider == "ollama":
        return OllamaClient()
    if provider == "fake":
        return FakeClient(model, data_dir)
    raise ValueError(f"unknown provider: {provider}")


class Cassette:
    def __init__(self, root: Path, mode: str):
        if mode not in {"record", "replay"}:
            raise ValueError("mode must be 'record' or 'replay'")
        self.root, self.mode = root, mode
        self.hits = self.misses = 0

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def call(self, request: dict, doc_id: str, client) -> tuple[dict, str]:
        key = request_key(request)
        path = self.path(key)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"], key
        if self.mode == "replay":
            raise CassetteMiss(
                f"No recording for {doc_id} (key {key[:12]}). The prompt, model, options or input text "
                f"changed since the last recording. Run the extraction in record mode against Ollama and commit "
                f"the new files in {self.root}."
            )
        self.misses += 1
        response = client.complete(request, doc_id)
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"doc_id": doc_id, "request_model": request["model"], "response": response}, indent=1) + "\n", encoding="utf-8")
        return response, key
