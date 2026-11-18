import json

import pytest

from empg.config import adapter_fingerprint, load_config


def test_shared_checkpoint_is_content_fingerprinted_and_changes_are_rejected(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    cfg = load_config(overrides=[f"train.initial_adapter={tmp_path}"])
    expected = cfg.train.initial_adapter_sha256
    assert expected == adapter_fingerprint(tmp_path)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="contents"):
        load_config(overrides=[f"train.initial_adapter={tmp_path}", f"train.initial_adapter_sha256={expected}"])


def test_incomplete_initialization_fails_before_loading_the_gpu_model(tmp_path):
    with pytest.raises(ValueError, match="saved LoRA"):
        load_config(overrides=[f"train.initial_adapter={tmp_path}"])


def test_modified_checkpoint_is_rejected_before_model_allocation(tmp_path, monkeypatch):
    from empg import trainer
    (tmp_path / "adapter_config.json").write_text("{}")
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    cfg = load_config(overrides=[f"train.initial_adapter={tmp_path}"])
    weights.write_bytes(b"modified")
    def unexpected(*args, **kwargs):
        raise AssertionError("model must not be allocated")
    monkeypatch.setattr(trainer, "LLMPolicy", unexpected)
    with pytest.raises(ValueError, match="initial adapter changed"):
        trainer.train(cfg)
