import torch
import json
import os
import pytest


@pytest.mark.parametrize("model_size", ["7b", "40b"])
def test_falcon_conversion(model_size):
    chkptpath = f"checkpoints/tiiuae/falcon-{model_size}/lit_model_finetuned.bin"

    if not os.path.isfile(chkptpath):
        pytest.skip(f"{chkptpath} not found")

    lit_model = torch.load(chkptpath)
    lit_keys = [k for k in lit_model.keys()]

    torch_model_path = f"checkpoints/tiiuae/falcon-{model_size}/pytorch_model.bin.index.json"

    with open(torch_model_path, "r") as json_file:
        hf_md = json.load(json_file)
    torch_keys = list(hf_md["weight_map"])

    assert all(litkey in torch_keys for litkey in lit_keys)


def test_copy_weights_open_llama():
    chkptpath = "checkpoints/openlm-research/open_llama_3b/lit_model_finetuned.bin"

    if not os.path.isfile(chkptpath):
        pytest.skip(f"{chkptpath} not found")

    lit_model = torch.load(chkptpath)
    lit_keys = [k for k in lit_model.keys()]

    torch_model_path = "checkpoints/openlm-research/open_llama_3b/pytorch_model.bin"
    torch_model = torch.load(torch_model_path)
    torch_keys = [k for k in torch_model.keys()]

    assert all(litkey in torch_keys for litkey in lit_keys)
