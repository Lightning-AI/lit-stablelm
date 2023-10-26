import json


def test_config():
    from lit_gpt import Config

    config = Config()
    assert config.name == ""
    assert config.block_size == 4096

    config = Config(block_size=2048)
    assert config.block_size == 2048

    config = Config.from_name("pythia-70m")
    assert config.block_size == 2048

    config = Config.from_name("pythia-70m", block_size=4096)
    assert config.block_size == 4096

    config = Config(hf_config={"name": "pythia-70m"})
    assert config.name == "pythia-70m"


def test_legacy_args(tmp_path):
    from lit_gpt import Config

    config = Config.from_name("pythia-70m", condense_ratio=2)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 2

    json_path = tmp_path / "config.json"
    with open(json_path, "w") as fp:
        json.dump({"condense_ratio": 3}, fp)

    config = Config.from_json(json_path)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 3
    config = Config.from_json(json_path, condense_ratio=2)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 2


def test_from_hf_name():
    from lit_gpt import Config

    # by short-hand name
    config0 = Config.from_name("tiny-llama-1.1b")
    # or by huggingface hub repo name
    config1 = Config.from_name("TinyLlama-1.1B-intermediate-step-480k-1T")
    assert config0 == config1
