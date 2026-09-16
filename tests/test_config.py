import pytest
from gpt.config import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    get_preset,
)


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.vocab_size == 50257
    assert cfg.n_embd == 768
    assert cfg.head_size == 64
    assert cfg.n_layer == 12


def test_model_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=100, n_head=7)

    with pytest.raises(ValueError, match="vocab_size"):
        ModelConfig(vocab_size=-1)


def test_model_config_presets():
    gpt2_cfg = get_preset("gpt2")
    assert gpt2_cfg.n_layer == 12
    assert gpt2_cfg.n_embd == 768

    nano_cfg = get_preset("nano_shakespeare")
    assert nano_cfg.vocab_size == 65
    assert nano_cfg.n_layer == 6

    with pytest.raises(KeyError):
        get_preset("non_existent_preset")


def test_training_config_batch_size():
    cfg = TrainingConfig(batch_size=16, grad_accum_steps=4)
    assert cfg.total_batch_size == 64


def test_serialization_roundtrip():
    cfg = ModelConfig(vocab_size=1000, n_embd=256, n_head=4, n_layer=4)
    json_str = cfg.to_json()
    cfg_loaded = ModelConfig.from_json(json_str)
    assert cfg_loaded.vocab_size == 1000
    assert cfg_loaded.n_embd == 256

    yaml_str = cfg.to_yaml()
    cfg_from_yaml = ModelConfig.from_yaml(yaml_str)
    assert cfg_from_yaml.vocab_size == 1000
    assert cfg_from_yaml.n_layer == 4


def test_inference_config():
    inf_cfg = InferenceConfig(temperature=0.7, top_k=40, top_p=0.9)
    assert inf_cfg.temperature == 0.7
    assert inf_cfg.top_k == 40
    assert inf_cfg.top_p == 0.9


def test_data_config():
    data_cfg = DataConfig(dataset_path="input.txt", train_split=0.85)
    assert data_cfg.dataset_path == "input.txt"
    assert data_cfg.train_split == 0.85
