from __future__ import annotations

import json

from labs.lab09_batch_processing.batch import build_batch_requests, parse_batch_output_jsonl, write_batch_jsonl
from labs.lab09_batch_processing.experiments import batch_output_to_results


def test_build_batch_requests_matches_groq_batch_schema():
    requests = build_batch_requests(["hi", "bye"], "llama-3.1-8b-instant")
    assert len(requests) == 2
    assert requests[0]["custom_id"] == "request-0"
    assert requests[0]["method"] == "POST"
    assert requests[0]["url"] == "/v1/chat/completions"
    assert requests[0]["body"]["model"] == "llama-3.1-8b-instant"
    assert requests[0]["body"]["messages"][0]["content"] == "hi"


def test_write_batch_jsonl_round_trips(tmp_path):
    requests = build_batch_requests(["a", "b", "c"], "llama-3.1-8b-instant")
    path = tmp_path / "batch_input.jsonl"
    write_batch_jsonl(requests, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["custom_id"] == "request-0"


def test_parse_batch_output_jsonl():
    raw = (
        '{"custom_id": "request-0", "response": {"status_code": 200, "body": {"usage": {"prompt_tokens": 10}}}, "error": null}\n'
        '{"custom_id": "request-1", "response": {"status_code": 200, "body": {"usage": {"prompt_tokens": 20}}}, "error": null}\n'
    )
    parsed = parse_batch_output_jsonl(raw)
    assert len(parsed) == 2
    assert parsed[1]["custom_id"] == "request-1"


def test_batch_output_to_results_extracts_usage_and_cost(config_with_pricing):
    parsed_lines = [
        {
            "custom_id": "request-0",
            "response": {
                "status_code": 200,
                "body": {"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
            },
            "error": None,
        }
    ]
    pricing = config_with_pricing.pricing["llama-3.1-8b-instant"]
    results = batch_output_to_results(parsed_lines, "llama-3.1-8b-instant", pricing)

    assert len(results) == 1
    assert results[0].tokens_and_cost.prompt_tokens == 100
    assert results[0].tokens_and_cost.estimated_cost_usd is not None
    assert results[0].metadata.execution_mode == "batch"


def test_batch_output_to_results_captures_error_message():
    parsed_lines = [{"custom_id": "request-0", "response": None, "error": {"message": "something failed"}}]
    results = batch_output_to_results(parsed_lines, "llama-3.1-8b-instant", None)
    assert results[0].metadata.error == "something failed"
    assert results[0].tokens_and_cost.prompt_tokens is None
