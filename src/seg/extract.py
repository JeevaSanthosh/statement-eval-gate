"""Run the extractor over every statement and save its predictions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .llm import OPTIONS, Cassette, make_client
from .schema import Statement


def inline_refs(schema: dict) -> dict:
    """Ollama turns the schema into a grammar; inlining $refs keeps that step simple."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def build_request(model: str, system_prompt: str, statement_text: str, think: bool | None = None) -> dict:
    schema = inline_refs(Statement.model_json_schema())
    # Ollama's guidance is to state the schema in the prompt as well as in `format`.
    system = f"{system_prompt.strip()}\n\nReturn JSON that matches this schema:\n{json.dumps(schema, indent=1)}"
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": statement_text},
        ],
        "format": schema,
        "options": OPTIONS,
    }
    if think is not None:  # only for models that can think; part of the request, so part of the cassette key
        request["think"] = think
    return request


def parse(content: str) -> tuple[dict, str | None]:
    """Validate a response against the schema. A response that fails counts as an empty answer."""
    try:
        return Statement.model_validate_json(content).model_dump(), None
    except ValidationError as exc:
        return {}, f"invalid response: {exc.error_count()} schema error(s)"


def run_extraction(
    data_dir: Path,
    prompt_path: Path,
    provider: str,
    model: str,
    out_dir: Path,
    mode: str = "replay",
    cassette_dir: Path = Path("cassettes"),
    limit: int | None = None,
    think: bool | None = None,
    verbose: bool = False,
) -> Path:
    # Explicit UTF-8 everywhere: Windows defaults to cp1252, which would garble '£' and
    # change every request hash, so recordings made on Windows would never replay in CI.
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))[:limit]
    prompt = prompt_path.read_text(encoding="utf-8")
    client = make_client(provider, model, data_dir)
    cassette = Cassette(cassette_dir, mode) if provider != "fake" else None

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for n, entry in enumerate(manifest, start=1):
            text = (data_dir / "text" / f"{entry['id']}.txt").read_text(encoding="utf-8")
            request = build_request(model, prompt, text, think)
            if cassette:
                recorded_before = cassette.misses
                response, key = cassette.call(request, entry["id"], client)
                source = "model" if cassette.misses > recorded_before else "recording"
            else:
                response, key, source = client.complete(request, entry["id"]), None, "fake"
            prediction, error = parse(response["content"])
            if verbose:
                took = f"{response['latency_ms'] / 1000:.0f}s" if response.get("latency_ms") else ""
                print(f"[{n}/{len(manifest)}] {entry['id']} {source} {took} {error or 'ok'}", flush=True)
            f.write(
                json.dumps(
                    {
                        "id": entry["id"],
                        "prediction": prediction,
                        "error": error,
                        "latency_ms": response.get("latency_ms"),
                        "prompt_tokens": response.get("prompt_tokens"),
                        "completion_tokens": response.get("completion_tokens"),
                        "cassette_key": key,
                    }
                )
                + "\n"
            )

    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "prompt": prompt_path.as_posix(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "options": OPTIONS,
                "think": think,
                "mode": mode,
                "documents": len(manifest),
                "recorded_now": cassette.misses if cassette else None,
                "seg_version": __version__,
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir
