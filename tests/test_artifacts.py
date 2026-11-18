import json

import pytest

from benchmarks.artifacts import read_jsonl, write_json


def test_interrupted_last_record_preserves_completed_observations(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"iteration": 1}\n{"iteration": 2')
    with pytest.warns(UserWarning, match="interrupted"):
        assert read_jsonl(path) == [{"iteration": 1}]
    path.write_text('{"iteration": 1}\ninvalid\n{"iteration": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_json_publication_is_complete_and_creates_its_directory(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    write_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert not path.with_suffix(".json.tmp").exists()
