# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import itertools
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from re import escape

import pytest
import torch
import yaml
from lightning import Fabric

from litgpt import Config
from litgpt.generate.sequentially import layer_to_device, replace_device, sequential
from litgpt.model import GPT, Block
from litgpt.scripts.download import download_from_hub
from tests.conftest import RunIf


@pytest.mark.parametrize(
    ("n_layer", "devices", "expected"),
    [
        (6, 1, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}),
        (6, 2, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}),
        (6, 3, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}),
        (6, 4, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}),
        (6, 5, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}),
        (6, 6, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}),
    ],
)
def test_layer_to_device(n_layer, devices, expected):
    with torch.device("meta"):
        model = GPT.from_name("pythia-14m", n_layer=n_layer)

    max_layers_per_device = math.ceil(n_layer / devices)
    actual = layer_to_device(model, Block, chunk_size=max_layers_per_device)
    expected = {f"transformer.h.{i}": v for i, v in expected.items()}
    assert actual == expected


def test_sequential_layer_to_device_mapping_not_possible():
    # Fewer layers than devices
    config = Config(n_layer=1)
    with torch.device("meta"):
        model = GPT(config)
    with pytest.raises(ValueError, match="number of layers in the model must be larger than the number of devices"):
        sequential(model, root=torch.device("cpu"), max_seq_length=128, devices=2)

    # Last device would get 0 layers
    config = Config(n_layer=6)
    with torch.device("meta"):
        model = GPT(config)
    with pytest.raises(RuntimeError, match="Not able to distribute the 6 layers across 4 devices"):
        sequential(model, root=torch.device("cpu"), max_seq_length=128, devices=4)


def path_to_device(model):
    return {k: str(v.device) for k, v in itertools.chain(model.named_parameters(), model.named_buffers())}


def test_replace_device():
    class Submodule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("foo", torch.tensor(1, device="cpu"))
            self.register_buffer("bar", torch.tensor(1, device="cpu"))

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modules = torch.nn.ModuleDict(
                {
                    "module1": torch.nn.Linear(1, 1, bias=True, device="meta"),
                    "module2": torch.nn.Linear(1, 1, bias=False, device="cpu"),
                }
            )
            self.submodule = Submodule()

    model = MyModel()
    assert path_to_device(model) == {
        "modules.module1.bias": "meta",
        "modules.module1.weight": "meta",
        "modules.module2.weight": "cpu",
        "submodule.bar": "cpu",
        "submodule.foo": "cpu",
    }
    model = replace_device(model, torch.device("cpu"), torch.device("meta"))
    assert path_to_device(model) == {
        "modules.module1.bias": "meta",
        "modules.module1.weight": "meta",
        "modules.module2.weight": "meta",
        "submodule.bar": "meta",
        "submodule.foo": "meta",
    }

    model = MyModel()
    model.submodule.bar = model.submodule.bar.to("meta")
    with pytest.raises(
        ValueError,
        match=escape("multiple devices: {'submodule.foo': device(type='cpu'), 'submodule.bar': device(type='meta')}"),
    ):
        replace_device(model, torch.device("cpu"), torch.device("meta"))


def _test_model_1device(accelerator):
    fabric = Fabric(accelerator=accelerator, devices=1)
    with torch.device("meta"):
        model = GPT.from_name("pythia-14m", n_layer=2)
        model.set_kv_cache(1)
    model = sequential(model, fabric.device, 15, 1)

    device_str = str(fabric.device)
    assert path_to_device(model) == {
        "cos": device_str,
        "sin": device_str,
        "lm_head.weight": device_str,
        "transformer.h.0.attn.qkv.bias": device_str,
        "transformer.h.0.attn.qkv.weight": device_str,
        "transformer.h.0.attn.proj.bias": device_str,
        "transformer.h.0.attn.proj.weight": device_str,
        "transformer.h.0.mlp.fc.bias": device_str,
        "transformer.h.0.mlp.fc.weight": device_str,
        "transformer.h.0.mlp.proj.bias": device_str,
        "transformer.h.0.mlp.proj.weight": device_str,
        "transformer.h.0.norm_1.bias": device_str,
        "transformer.h.0.norm_1.weight": device_str,
        "transformer.h.0.norm_2.bias": device_str,
        "transformer.h.0.norm_2.weight": device_str,
        "transformer.h.0.attn.kv_cache.k": device_str,
        "transformer.h.0.attn.kv_cache.v": device_str,
        "transformer.h.1.attn.qkv.bias": device_str,
        "transformer.h.1.attn.qkv.weight": device_str,
        "transformer.h.1.attn.proj.bias": device_str,
        "transformer.h.1.attn.proj.weight": device_str,
        "transformer.h.1.mlp.fc.bias": device_str,
        "transformer.h.1.mlp.fc.weight": device_str,
        "transformer.h.1.mlp.proj.bias": device_str,
        "transformer.h.1.mlp.proj.weight": device_str,
        "transformer.h.1.norm_1.bias": device_str,
        "transformer.h.1.norm_1.weight": device_str,
        "transformer.h.1.norm_2.bias": device_str,
        "transformer.h.1.norm_2.weight": device_str,
        "transformer.h.1.attn.kv_cache.k": device_str,
        "transformer.h.1.attn.kv_cache.v": device_str,
        "transformer.ln_f.bias": device_str,
        "transformer.ln_f.weight": device_str,
        "transformer.wte.weight": device_str,
    }
    assert model.max_seq_length == 15


@RunIf(min_cuda_gpus=1)
def test_model_1device_cuda():
    _test_model_1device("cuda")


def test_model_1device_cpu():
    _test_model_1device("cpu")


def find_forward_hooks(module):
    mapping = defaultdict(list)
    for name, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            hook_data = ("forward_pre_hook", hook.func.__name__, hook.args, hook.keywords)
            mapping[name].append(hook_data)
        for hook in submodule._forward_hooks.values():
            hook_data = ("forward_hook", hook.func.__name__, hook.args, hook.keywords)
            mapping[name].append(hook_data)
    return dict(mapping)


@RunIf(min_cuda_gpus=2)
def test_model_forward_hooks():
    fabric = Fabric(accelerator="cuda", devices=1)
    with torch.device("meta"):
        model = GPT.from_name("pythia-14m")  # 6 layers
    model = sequential(model, fabric.device, max_seq_length=15, devices=2)

    hooks = find_forward_hooks(model)
    actual = path_to_device(model)
    assert actual == {
        "lm_head.weight": "cuda:0",
        "transformer.wte.weight": "cuda:0",
        "transformer.h.0.norm_1.weight": "cuda:0",
        "transformer.h.0.norm_1.bias": "cuda:0",
        "transformer.h.0.attn.qkv.weight": "cuda:0",
        "transformer.h.0.attn.qkv.bias": "cuda:0",
        "transformer.h.0.attn.proj.weight": "cuda:0",
        "transformer.h.0.attn.proj.bias": "cuda:0",
        "transformer.h.0.norm_2.weight": "cuda:0",
        "transformer.h.0.norm_2.bias": "cuda:0",
        "transformer.h.0.mlp.fc.weight": "cuda:0",
        "transformer.h.0.mlp.fc.bias": "cuda:0",
        "transformer.h.0.mlp.proj.weight": "cuda:0",
        "transformer.h.0.mlp.proj.bias": "cuda:0",
        "transformer.h.1.norm_1.weight": "cuda:0",
        "transformer.h.1.norm_1.bias": "cuda:0",
        "transformer.h.1.attn.qkv.weight": "cuda:0",
        "transformer.h.1.attn.qkv.bias": "cuda:0",
        "transformer.h.1.attn.proj.weight": "cuda:0",
        "transformer.h.1.attn.proj.bias": "cuda:0",
        "transformer.h.1.norm_2.weight": "cuda:0",
        "transformer.h.1.norm_2.bias": "cuda:0",
        "transformer.h.1.mlp.fc.weight": "cuda:0",
        "transformer.h.1.mlp.fc.bias": "cuda:0",
        "transformer.h.1.mlp.proj.weight": "cuda:0",
        "transformer.h.1.mlp.proj.bias": "cuda:0",
        "transformer.h.2.norm_1.weight": "cuda:0",
        "transformer.h.2.norm_1.bias": "cuda:0",
        "transformer.h.2.attn.qkv.weight": "cuda:0",
        "transformer.h.2.attn.qkv.bias": "cuda:0",
        "transformer.h.2.attn.proj.weight": "cuda:0",
        "transformer.h.2.attn.proj.bias": "cuda:0",
        "transformer.h.2.norm_2.weight": "cuda:0",
        "transformer.h.2.norm_2.bias": "cuda:0",
        "transformer.h.2.mlp.fc.weight": "cuda:0",
        "transformer.h.2.mlp.fc.bias": "cuda:0",
        "transformer.h.2.mlp.proj.weight": "cuda:0",
        "transformer.h.2.mlp.proj.bias": "cuda:0",
        "transformer.h.3.norm_1.weight": "cuda:1",
        "transformer.h.3.norm_1.bias": "cuda:1",
        "transformer.h.3.attn.qkv.weight": "cuda:1",
        "transformer.h.3.attn.qkv.bias": "cuda:1",
        "transformer.h.3.attn.proj.weight": "cuda:1",
        "transformer.h.3.attn.proj.bias": "cuda:1",
        "transformer.h.3.norm_2.weight": "cuda:1",
        "transformer.h.3.norm_2.bias": "cuda:1",
        "transformer.h.3.mlp.fc.weight": "cuda:1",
        "transformer.h.3.mlp.fc.bias": "cuda:1",
        "transformer.h.3.mlp.proj.weight": "cuda:1",
        "transformer.h.3.mlp.proj.bias": "cuda:1",
        "transformer.h.4.norm_1.weight": "cuda:1",
        "transformer.h.4.norm_1.bias": "cuda:1",
        "transformer.h.4.attn.qkv.weight": "cuda:1",
        "transformer.h.4.attn.qkv.bias": "cuda:1",
        "transformer.h.4.attn.proj.weight": "cuda:1",
        "transformer.h.4.attn.proj.bias": "cuda:1",
        "transformer.h.4.norm_2.weight": "cuda:1",
        "transformer.h.4.norm_2.bias": "cuda:1",
        "transformer.h.4.mlp.fc.weight": "cuda:1",
        "transformer.h.4.mlp.fc.bias": "cuda:1",
        "transformer.h.4.mlp.proj.weight": "cuda:1",
        "transformer.h.4.mlp.proj.bias": "cuda:1",
        "transformer.h.5.norm_1.weight": "cuda:1",
        "transformer.h.5.norm_1.bias": "cuda:1",
        "transformer.h.5.attn.qkv.weight": "cuda:1",
        "transformer.h.5.attn.qkv.bias": "cuda:1",
        "transformer.h.5.attn.proj.weight": "cuda:1",
        "transformer.h.5.attn.proj.bias": "cuda:1",
        "transformer.h.5.norm_2.weight": "cuda:1",
        "transformer.h.5.norm_2.bias": "cuda:1",
        "transformer.h.5.mlp.fc.weight": "cuda:1",
        "transformer.h.5.mlp.fc.bias": "cuda:1",
        "transformer.h.5.mlp.proj.weight": "cuda:1",
        "transformer.h.5.mlp.proj.bias": "cuda:1",
        "transformer.ln_f.weight": "cuda:0",
        "transformer.ln_f.bias": "cuda:0",
        "cos": "cuda:0",
        "sin": "cuda:0",
        "transformer.h.0.attn.kv_cache.k": "cuda:0",
        "transformer.h.0.attn.kv_cache.v": "cuda:0",
        "transformer.h.1.attn.kv_cache.k": "cuda:0",
        "transformer.h.1.attn.kv_cache.v": "cuda:0",
        "transformer.h.2.attn.kv_cache.k": "cuda:0",
        "transformer.h.2.attn.kv_cache.v": "cuda:0",
        "transformer.h.3.attn.kv_cache.k": "cuda:1",
        "transformer.h.3.attn.kv_cache.v": "cuda:1",
        "transformer.h.4.attn.kv_cache.k": "cuda:1",
        "transformer.h.4.attn.kv_cache.v": "cuda:1",
        "transformer.h.5.attn.kv_cache.k": "cuda:1",
        "transformer.h.5.attn.kv_cache.v": "cuda:1",
    }
    assert hooks == {
        "transformer.h.3": [("forward_pre_hook", "move_block_input", (torch.device(type="cuda", index=1),), {})],
        "transformer.h.4": [("forward_pre_hook", "move_block_input", (torch.device(type="cuda", index=1),), {})],
        "transformer.h.5": [
            ("forward_pre_hook", "move_block_input", (torch.device(type="cuda", index=1),), {}),
            ("forward_hook", "move_block_output", (torch.device(type="cuda", index=0),), {}),
        ],
    }


root = Path(__file__).parent.parent.resolve()


@RunIf(min_cuda_gpus=2)
def test_base_with_sequentially(tmp_path):
    # download the tokenizer
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)
    checkpoint_dir = tmp_path / "EleutherAI/pythia-14m"
    # save the config
    config = Config.from_name("pythia-14m")
    (checkpoint_dir / "model_config.yaml").write_text(yaml.dump(asdict(config)))
    # create a state dict to load from
    torch.save(GPT(config).state_dict(), checkpoint_dir / "lit_model.pth")

    args = [
        str(checkpoint_dir),
        "--num_samples=1",
        "--max_new_tokens=10",
        "--precision=16-true",
        "--temperature=0.0",
    ]
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    sequential_stdout = subprocess.check_output(
        [sys.executable, "-m", "litgpt", "generate_sequentially", *args], env=env, cwd=root,
    ).decode()

    assert "What food do llamas eat?" in sequential_stdout


def test_cli():
    args = ["litgpt", "generate_sequentially", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Generation script that partitions layers across" in output
