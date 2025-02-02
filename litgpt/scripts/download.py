# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import shutil
from typing import List, Optional, Tuple

import torch
from lightning_utilities.core.imports import RequirementCache

from litgpt.config import configs
from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")
_HF_TRANSFER_AVAILABLE = RequirementCache("hf_transfer")


def download_from_hub(
    repo_id: str,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    tokenizer_only: bool = False,
    convert_checkpoint: bool = True,
    dtype: Optional[str] = None,
    checkpoint_dir: Path = Path("checkpoints"),
    model_name: Optional[str] = None,
) -> None:
    """Download weights or tokenizer data from the Hugging Face Hub.

    Arguments:
        repo_id: The repository ID in the format ``org/name`` or ``user/name`` as shown in Hugging Face.
            If "list" is provided as input, a list of the currently supported models in LitGPT and quits.
        access_token: Optional API token to access models with restrictions.
        tokenizer_only: Whether to download only the tokenizer files.
        convert_checkpoint: Whether to convert the checkpoint files to the LitGPT format after downloading.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to use for this repo_id. This is useful to download alternative weights of
            existing architectures.
    """
    options = [f"{config['hf_config']['org']}/{config['hf_config']['name']}" for config in configs]

    if repo_id == "list":
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(sorted(options, key=lambda x: x.lower())))
        return

    if model_name is None and repo_id not in options:
        print(f"Unsupported `repo_id`: {repo_id}."
        "\nIf you are trying to download alternative "
        "weights for a supported model, please specify the corresponding model via the `--model_name` option, "
        "for example, `litgpt download NousResearch/Hermes-2-Pro-Llama-3-8B --model_name Llama-3-8B`."
        "\nAlternatively, please choose a valid `repo_id` from the list of supported models, which can be obtained via "
        "`litgpt download list`.")
        return

    from huggingface_hub import snapshot_download
    if importlib.util.find_spec("hf_transfer") is None:
        print("It is recommended to install hf_transfer for faster checkpoint download speeds: `pip install hf_transfer`")

    download_files = ["tokenizer*", "generation_config.json", "config.json"]
    if not tokenizer_only:
        bins, safetensors, info = find_weight_files(repo_id, access_token)

        total_weight_size_bytes = 0
        if bins:
            total_weight_size_bytes = sum(
                (file.size or 0)
                for file in info.siblings
                if file.rfilename.endswith(".bin") or file.rfilename.endswith(".bin.index.json")
            )
        elif safetensors:
            total_weight_size_bytes = sum(
                (file.size or 0)
                for file in info.siblings
                if file.rfilename.endswith(".safetensors")
            )
        else:
            raise ValueError(f"Couldn't find weight files for {repo_id}")

        weight_size_gb = total_weight_size_bytes / (1024**3)
        free_space_bytes = shutil.disk_usage(str(checkpoint_dir)).free
        free_space_gb = free_space_bytes / (1024**3)

        # 2x because we create lit_model.pth before deleting the downloaded weights,
        # so we intermittenly have 2 sets of weights on disk
        if weight_size_gb > 2*free_space_gb:
            if os.getenv("LIGHTNING_CLOUD_SPACE_ID") is not None:
                studio_text = " Please switch to a larger Studio with more disk space."
            else:
                studio_text = ""
            raise RuntimeError(
                f"Not enough disk space to download {repo_id} weights. "
                f"Needed: ~{2*weight_size_gb:.2f} GB, free: ~{free_space_gb:.2f} GB.{studio_text}"
            )

        if bins:
            # covers `.bin` files and `.bin.index.json`
            download_files.append("*.bin*")
        elif safetensors:
            if not _SAFETENSORS_AVAILABLE:
                raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
            download_files.append("*.safetensors*")
        else:
            raise ValueError(f"Couldn't find weight files for {repo_id}")

    import huggingface_hub._snapshot_download as download
    import huggingface_hub.constants as constants

    previous = constants.HF_HUB_ENABLE_HF_TRANSFER
    if _HF_TRANSFER_AVAILABLE and not previous:
        print("Setting HF_HUB_ENABLE_HF_TRANSFER=1")
        constants.HF_HUB_ENABLE_HF_TRANSFER = True
        download.HF_HUB_ENABLE_HF_TRANSFER = True

    directory = checkpoint_dir / repo_id
    with gated_repo_catcher(repo_id, access_token):
        snapshot_download(
            repo_id,
            local_dir=directory,
            allow_patterns=download_files,
            token=access_token,
        )

    constants.HF_HUB_ENABLE_HF_TRANSFER = previous
    download.HF_HUB_ENABLE_HF_TRANSFER = previous

    if convert_checkpoint and not tokenizer_only:
        print("Converting checkpoint files to LitGPT format.")
        convert_hf_checkpoint(checkpoint_dir=directory, dtype=dtype, model_name=model_name)


def find_weight_files(repo_id: str, access_token: Optional[str]) -> Tuple[List[str], List[str]]:
    from huggingface_hub import repo_info
    from huggingface_hub.utils import filter_repo_objects

    with gated_repo_catcher(repo_id, access_token):
        info = repo_info(repo_id, token=access_token, files_metadata=True)
    filenames = [f.rfilename for f in info.siblings]
    bins = list(filter_repo_objects(items=filenames, allow_patterns=["*model*.bin*"]))
    safetensors = list(filter_repo_objects(items=filenames, allow_patterns=["*.safetensors*"]))
    return bins, safetensors, info


@contextmanager
def gated_repo_catcher(repo_id: str, access_token: Optional[str]):
    try:
        yield
    except OSError as e:
        err_msg = str(e)
        if "Repository Not Found" in err_msg:
            raise ValueError(
                f"Repository at https://huggingface.co/api/models/{repo_id} not found."
                " Please make sure you specified the correct `repo_id`."
            ) from None
        elif "gated repo" in err_msg:
            if not access_token:
                raise ValueError(
                    f"https://huggingface.co/{repo_id} requires authentication, please set the `HF_TOKEN=your_token`"
                    " environment variable or pass `--access_token=your_token`. You can find your token by visiting"
                    " https://huggingface.co/settings/tokens."
                ) from None
            else:
                raise ValueError(
                    f"https://huggingface.co/{repo_id} requires authentication. The access token provided by `HF_TOKEN=your_token`"
                    " environment variable or `--access_token=your_token` may not have sufficient access rights. Please"
                    f" visit https://huggingface.co/{repo_id} for more information."
                ) from None
        raise e from None
