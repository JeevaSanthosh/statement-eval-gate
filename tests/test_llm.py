import pytest

from seg.extract import build_request, parse
from seg.llm import Cassette, CassetteMiss, request_key


class CountingClient:
    def __init__(self):
        self.calls = 0

    def complete(self, request, doc_id):
        self.calls += 1
        return {"content": '{"provider": "X"}', "latency_ms": 5}


def test_record_then_replay_returns_the_recorded_response(tmp_path):
    client = CountingClient()
    request = build_request("m", "prompt", "statement text")
    recorded, key = Cassette(tmp_path, "record").call(request, "d1", client)
    replayed, key2 = Cassette(tmp_path, "replay").call(request, "d1", client=None)
    assert recorded == replayed and key == key2
    assert client.calls == 1


def test_record_mode_reuses_existing_recordings(tmp_path):
    client = CountingClient()
    request = build_request("m", "prompt", "statement text")
    Cassette(tmp_path, "record").call(request, "d1", client)
    Cassette(tmp_path, "record").call(request, "d1", client)
    assert client.calls == 1


def test_replay_refuses_a_changed_prompt(tmp_path):
    Cassette(tmp_path, "record").call(build_request("m", "prompt v1", "text"), "d1", CountingClient())
    with pytest.raises(CassetteMiss, match="re-record|record mode"):
        Cassette(tmp_path, "replay").call(build_request("m", "prompt v2", "text"), "d1", client=None)


def test_key_depends_on_model_and_input():
    base = request_key(build_request("m", "p", "t"))
    assert base != request_key(build_request("other", "p", "t"))
    assert base != request_key(build_request("m", "p", "t2"))


def test_think_setting_is_sent_and_changes_the_key():
    assert "think" not in build_request("m", "p", "t")
    off = build_request("m", "p", "t", think=False)
    assert off["think"] is False
    assert request_key(off) != request_key(build_request("m", "p", "t", think=True))


def test_request_schema_has_no_unresolved_refs():
    assert "$ref" not in str(build_request("m", "p", "t")["format"])


def test_invalid_response_is_scored_as_empty():
    prediction, error = parse('{"holdings": "not a list"}')
    assert prediction == {} and error
