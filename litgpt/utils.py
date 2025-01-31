# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Utility functions for training and inference."""
import inspect
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
from dataclasses import asdict, is_dataclass
from io import BytesIO
from packaging import version
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Mapping, Optional, TypeVar, Union
import warnings

import lightning as L
import torch
import torch.nn as nn
import torch.utils._device
import yaml
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import instantiate_class
from torch.serialization import normalize_storage_type
from typing_extensions import Self


if TYPE_CHECKING:
    from litgpt import GPT, Config


def init_out_dir(out_dir: Path) -> Path:
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    if not out_dir.is_absolute() and "LIGHTNING_ARTIFACTS_DIR" in os.environ:
        return Path(os.getenv("LIGHTNING_ARTIFACTS_DIR")) / out_dir
    return out_dir


def find_resume_path(resume: Union[bool, Literal["auto"], Path], out_dir: Path) -> Optional[Path]:
    if not resume or isinstance(resume, Path):
        return resume

    resume_path = max(out_dir.rglob("step-*/*.pth"), key=(lambda p: int(p.parent.name.split("-")[1])), default=None)
    if resume == "auto":
        return resume_path
    if resume is True and resume_path is None:
        raise FileNotFoundError(
            f"You passed `--resume=True`, but no checkpont file was found in `--out_dir={out_dir}`."
        )
    return resume_path


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)
            else:
                total += p.numel()
    return total


def reset_parameters(module: nn.Module) -> None:
    """Calls `reset_parameters` on the module and all its submodules."""
    for mod in module.modules():
        if callable(getattr(mod, "reset_parameters", None)):
            mod.reset_parameters()


def check_valid_checkpoint_dir(
        checkpoint_dir: Path,
        model_filename: str = "lit_model.pth",
        verbose: bool = True,
        raise_error: bool = False,
        ignore_tokenizer_files: bool = False
    ) -> None:

    files = {
        model_filename: (checkpoint_dir / model_filename).is_file(),
        "model_config.yaml": (checkpoint_dir / "model_config.yaml").is_file(),
    }
    if not ignore_tokenizer_files:
        files.update({
            "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file() or
                                                (checkpoint_dir / "tokenizer.model").is_file(),
            "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
        })

    if checkpoint_dir.is_dir():
        if all(files.values()):
            # we're good
            return
        problem = f" is missing the files: {[f for f, exists in files.items() if not exists]!r}"
    else:
        problem = " is not a checkpoint directory"

    # list locally available checkpoints
    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = "\n".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    if verbose:
        error_message = (
            f"checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}."
            "\nFind download instructions at https://github.com/Lightning-AI/litgpt/blob/main/tutorials\n"
            f"{extra}\nSee all download options by running:\n litgpt download"
        )
        print(error_message, file=sys.stderr)

    if raise_error:
        raise FileNotFoundError(f"checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}.")
    else:
        raise SystemExit(1)


class SavingProxyForStorage:
    def __init__(self, obj, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.saver = saver
        if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
            raise TypeError(f"expected storage, not {type(obj)}")

        # this logic is taken from PyTorch 2.0+ torch/serialization.py
        if isinstance(obj, torch.storage.TypedStorage):
            # PT upstream wants to deprecate this eventually...
            storage = obj._untyped_storage
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()
        else:
            storage = obj
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        storage_key = saver._write_storage_and_return_key(storage)
        location = torch.serialization.location_tag(storage)

        self.storage_info = ("storage", storage_type, storage_key, location, storage_numel)

    def __reduce_ex__(self, protocol_version):
        assert False, "this should be handled with out of band"


class SavingProxyForTensor:
    def __init__(self, tensor, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.reduce_ret_fn, reduce_args = tensor.__reduce_ex__(protocol_version)
        if reduce_args[0] == torch._utils._rebuild_tensor_v2:
            # for Tensors with Python attributes
            (a0, a1, (storage, *a2_other), *other_reduce_args) = reduce_args
            assert isinstance(storage, (torch.storage.TypedStorage, torch.storage.UntypedStorage)), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (a0, a1, (storage_proxy, *a2_other), *other_reduce_args)
        else:
            (storage, *other_reduce_args) = reduce_args
            assert isinstance(storage, (torch.storage.TypedStorage, torch.storage.UntypedStorage)), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (storage_proxy, *other_reduce_args)

    def __reduce_ex__(self, protocol_version):
        if protocol_version != self.protocol_version:
            raise RuntimeError(f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}")
        return self.reduce_ret_fn, self.reduce_args


class IncrementalPyTorchPickler(pickle.Pickler):
    def __init__(self, saver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dtypes = {}
        self.saver = saver
        self.id_map = {}

    # this logic is taken from PyTorch 2.0+ torch/serialization.py
    def persistent_id(self, obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, SavingProxyForStorage):
            return obj.storage_info

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in self.storage_dtypes:
                    if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that view the same data as different types"
                        )
                else:
                    self.storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = self.id_map.get(storage._cdata)
            if storage_key is None:
                storage_key = self.saver._write_storage_and_return_key(storage)
                self.id_map[storage._cdata] = storage_key
            location = torch.serialization.location_tag(storage)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None


class incremental_save:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0
        self.protocol_version = 2

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self, protocol_version=self.protocol_version)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        # Write the pickle data for `obj`
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=self.protocol_version)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()

        current_version = version.parse(torch.__version__)
        threshold_version = version.parse("2.2.2")
        if current_version <= threshold_version:
            self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        else:
            self.zipfile.write_record(name, storage, num_bytes)

        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()


T = TypeVar("T")


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        # See [non_masked_elems div note]
        return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    non_masked_elems = (targets != ignore_index).sum()
    # [non_masked_elems div note]:
    #   max(1, non_masked_elems) would be more ergonomic to avoid a division by zero. However that
    #   results in a python int which is then passed back to torch division. By using the
    #   `x.maximum(torch.ones_like(x))` pattern we avoid a cudaStreamSynchronize.
    return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str) -> Dict:
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
    return state_dict


def get_default_supported_precision(training: bool) -> str:
    """
    Return the default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: If True, returns '-mixed' version of the precision; if False, returns '-true' version.

    Returns:
        The default precision that is suitable for the task and is supported by the hardware.
    """
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed" if training else "bf16-true"
        else:
            return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def load_checkpoint(fabric: L.Fabric, model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)


def flops_per_param(max_seq_length: int, n_layer: int, n_embd: int, n_params: int) -> int:
    flops_per_token = 2 * n_params  # each parameter is used for a MAC (2 FLOPS) per network operation
    # this assumes that all samples have a fixed length equal to the block size
    # which is most likely false during finetuning
    flops_per_seq = flops_per_token * max_seq_length
    attn_flops_per_seq = n_layer * 2 * 2 * (n_embd * (max_seq_length**2))
    return flops_per_seq + attn_flops_per_seq


def estimate_flops(model: "GPT", training: bool) -> int:
    """Measures estimated FLOPs for MFU.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    # using all parameters for this is a naive over estimation because not all model parameters actually contribute to
    # this FLOP computation (e.g. embedding, norm). For this reason, the result will be higher by a fixed percentage
    # (~10%) compared to the measured FLOPs, making those lower but more realistic.
    # For a proper estimate, this needs a more fine-grained calculation as in Appendix A of the paper.
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(
        model.max_seq_length, model.config.n_layer, model.config.n_embd, n_trainable_params
    )
    # forward + backward + gradients (assumes no gradient accumulation)
    ops_per_step = 3 if training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(model.max_seq_length, model.config.n_layer, model.config.n_embd, n_frozen_params)
    # forward + backward
    frozen_ops_per_step = 2 if training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


def copy_config_files(source_dir: Path, out_dir: Path) -> None:
    """Copies the specified configuration and tokenizer files into the output directory."""

    config_files = ["config.json", "generation_config.json", "model_config.yaml"]
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]

    for file_name in config_files + tokenizer_files:
        src_path = source_dir / file_name
        if src_path.exists():
            shutil.copy(src_path, out_dir)


def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI, set_config_read_mode, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    return CLI(*args, **kwargs)


def capture_hparams() -> Dict[str, Any]:
    """Captures the local variables ('hyperparameters') from where this function gets called."""
    caller_frame = inspect.currentframe().f_back
    locals_of_caller = caller_frame.f_locals
    hparams = {}
    for name, value in locals_of_caller.items():
        if value is None or isinstance(value, (int, float, str, bool, Path)):
            hparams[name] = value
        elif is_dataclass(value):
            hparams[name] = asdict(value)
        else:
            hparams[name] = str(value)
    return hparams


def save_hyperparameters(function: callable, checkpoint_dir: Path) -> None:
    """Captures the CLI parameters passed to `function` without running `function` and saves them to the checkpoint."""
    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    known_commands = [
        ("finetune_full",),  # For subcommands, use `("finetune", "full")` etc
        ("finetune_lora",),
        ("finetune_adapter",),
        ("finetune_adapter_v2",),
        ("finetune",),
        ("pretrain",),
    ]
    for known_command in known_commands:
        unwanted = slice(1, 1 + len(known_command))
        if tuple(sys.argv[unwanted]) == known_command:
            sys.argv[unwanted] = []

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)


def save_config(config: "Config", checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def parse_devices(devices: Union[str, int]) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    resume: Optional[bool] = None,
    **kwargs: Any,
):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project=name, resume=resume, **kwargs)
    raise ValueError(f"`--logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'.")


def get_argument_names(cls):
    sig = inspect.signature(cls.__init__)
    return {name for name, param in sig.parameters.items()
            if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]}


def instantiate_bnb_optimizer(optimizer, model_parameters):
    if (isinstance(optimizer, str) and "AdamW" not in optimizer) or (isinstance(optimizer, dict) and "AdamW" not in optimizer.get("class_path", "")):
        raise ValueError("The chosen quantization format only supports the AdamW optimizer.")

    import bitsandbytes as bnb
    if isinstance(optimizer, str):
        optimizer = bnb.optim.PagedAdamW(model_parameters)
    else:
        optim_args = get_argument_names(bnb.optim.PagedAdamW)
        allowed_kwargs = {key: optimizer["init_args"][key] for key in optim_args & optimizer["init_args"].keys()}
        optimizer = bnb.optim.PagedAdamW(model_parameters, **allowed_kwargs)
    return optimizer


def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    # Special care taken where some optimizers do not have some parameters referenced in some of the code, for example "fused" in the pretrain.py script:
    #   bnb.optim.AdamW8bit
    #   grokadamw.GrokAdamW
    #   torch.optim.RMSprop

    if isinstance(optimizer, str):
        if "." in optimizer:
            class_module, class_name = optimizer.rsplit(".", 1)
        else:
            class_module, class_name = "torch.optim", optimizer

        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}
        optimizer = optimizer_cls(model_parameters, **kwargs)
    elif isinstance(optimizer, dict):
        optimizer = dict(optimizer)
        class_module, class_name = optimizer["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}

        optimizer["init_args"].update(kwargs)
        optimizer = instantiate_class(model_parameters, optimizer)
    else:
        raise ValueError(f'Unrecognized "optimizer" value: {optimizer}')

    return optimizer


def extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (not checkpoint_dir.is_dir() and
                             checkpoint_dir.parts[0] != "checkpoints" and
                             not checkpoint_dir.is_absolute() and
                             new_checkpoint_dir.exists())
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dir


def check_file_size_on_cpu_and_warn(checkpoint_path, device, size_limit=4_509_715_660):
    """
    Checks the file size and raises a warning if it exceeds the size_limit.
    The default size limit is 4.2 GB, the size of TinyLlama 1.1B: 4.2 * 1024 * 1024 * 1024 = 4_509_715_660
    """
    size = 0.0
    if os.path.exists(checkpoint_path):
        size = os.path.getsize(checkpoint_path)
        if size > size_limit and str(device) == "cpu":
            warnings.warn(
                f"The file size of {checkpoint_path} is over {size_limit/1024/1024/1024:.1f} GB. Using a model "
                "with more than 1B parameters on a CPU can be slow, it is recommended to switch to a GPU."
            )
    return size


def auto_download_checkpoint(model_name, access_token=None, ignore_tokenizer_files=False):
    from litgpt.scripts.download import download_from_hub  # moved here due to circular import issue

    checkpoint_dir = extend_checkpoint_dir(Path(model_name))
    try:
        check_valid_checkpoint_dir(checkpoint_dir, verbose=False, raise_error=True, ignore_tokenizer_files=ignore_tokenizer_files)
    except FileNotFoundError as e:
        if access_token is None:
            access_token = os.getenv("HF_TOKEN")

        if checkpoint_dir.parts[0] != "checkpoints" and not checkpoint_dir.is_absolute():
            download_from_hub(repo_id=str(model_name), access_token=access_token)
            checkpoint_dir = Path("checkpoints") / checkpoint_dir
        else:
            raise e

    return checkpoint_dir


def check_nvlink_connectivity(fabric=None):
    """Checks GPU connectivity for both NVIDIA and AMD GPUs.

    This function delegates to vendor-specific implementations based on
    the detected GPU vendor.
    """
    if fabric is not None:
        custom_print = fabric.print
    else:
        custom_print = print

    if os.getenv("RANK", "0") == "0":
        try:
            if torch.cuda.is_available():
                device_properties = torch.cuda.get_device_properties(0)
                gpu_name = device_properties.name.lower()
                if "nvidia" in gpu_name:
                    _check_nvidia_connectivity(custom_print)
                elif "advanced micro devices" in gpu_name or "amd" in gpu_name:
                    _check_amd_connectivity(custom_print)
                else:
                    custom_print(f"Unrecognized GPU vendor: {device_properties.name}")
            else:
                custom_print("No GPUs available")
        except Exception as e:
            custom_print(f"An error occurred while checking GPU connectivity: {e}")


def _check_nvidia_connectivity(custom_print):
    """Checks NVLink connectivity on NVIDIA GPUs."""
    result = subprocess.run(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("Failed to run nvidia-smi")
        return

    lines = result.stdout.strip().split("\n")
    start_index = next((i for i, line in enumerate(lines) if "GPU0" in line), None)
    if start_index is None:
        custom_print("Failed to parse nvidia-smi output")
        return

    headers_line = lines[start_index]
    headers = headers_line.split()
    gpu_regex = re.compile(r"^GPU\d+$")
    gpu_count = len([header for header in headers if gpu_regex.match(header)])

    all_nvlink = True
    for line in lines[start_index + 1 : start_index + 1 + gpu_count]:
        columns = line.split()
        connections = columns[1 : 1 + gpu_count]
        if not all("NV" in conn for conn in connections if conn != "X"):
            all_nvlink = False
            break

    if all_nvlink:
        custom_print("All GPUs are fully connected via NVLink.")
    else:
        custom_print(
            "Warning: Not all GPUs are fully connected via NVLink. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )


def _check_amd_connectivity(custom_print):
    """Checks XGMI connectivity on AMD GPUs."""
    result = subprocess.run(["rocm-smi", "--showtopotype"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("Failed to run rocm-smi")
        return

    lines = result.stdout.strip().split("\n")
    gpu_header_index = next((i for i, line in enumerate(lines) if re.match(r"^\s*GPU0", line)), None)
    if gpu_header_index is None or gpu_header_index == 0:
        custom_print("Failed to parse rocm-smi output (no GPU headers found)")
        return

    header_line = lines[gpu_header_index - 1]
    headers = header_line.strip().split()
    gpu_regex = re.compile(r"^GPU\d+$")
    gpu_count = len([header for header in headers if gpu_regex.match(header)])

    gpu_lines = []
    for line in lines[gpu_header_index : gpu_header_index + gpu_count]:
        if re.match(r"^\s*GPU\d+", line):
            gpu_lines.append(line.strip())
    if len(gpu_lines) != gpu_count:
        custom_print("Mismatch in GPU count when parsing rocm-smi output")
        return

    all_xgmi = True
    for line in gpu_lines:
        columns = line.split()
        connections = columns[1 : 1 + gpu_count]
        for conn in connections:
            if conn not in ("XGMI", "0"):
                all_xgmi = False
                break
        if not all_xgmi:
            break

    if all_xgmi:
        custom_print("All GPUs are fully connected via XGMI.")
    else:
        custom_print(
            "Warning: Not all GPUs are fully connected via XGMI. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )


def fix_and_load_json(s):
    # Remove trailing commas before } or ]
    s = re.sub(r',(\s*[}\]])', r'\1', s)

    # Insert missing commas between properties
    # Match positions where a value is followed by a newline and then a quote without a comma
    pattern = r'(?<=[}\]0-9truefalsenull"])\s*(\n\s*)"'
    replacement = r',\1"'
    s = re.sub(pattern, replacement, s)

    # Now try to parse the JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after fixing: {e}")


def create_finetuning_performance_report(training_time, token_counts, device_type):
    tok_sec = token_counts["raw_tokens_plus_prompt_template_and_padding"] / training_time
    output = f"""
| ------------------------------------------------------
| Token Counts
| - Input Tokens              :  {token_counts["raw_tokens"]:>5}
| - Tokens w/ Prompt          :  {token_counts["raw_tokens_plus_prompt_template"]:>5}
| - Total Tokens (w/ Padding) :  {token_counts["raw_tokens_plus_prompt_template_and_padding"]:>5}
| -----------------------------------------------------
| Performance
| - Training Time             :  {training_time:.2f} s
| - Tok/sec                   :  {tok_sec:.2f} tok/s
| -----------------------------------------------------
"""

    if device_type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        output += f"| Memory Usage                                                                 \n"
        output += f"| - Memory Used               :  {memory_used:.02f} GB                                        \n"
    output += "-------------------------------------------------------\n"

    return output


def select_sft_generate_example(eval, data):

    if eval.evaluate_example == "first":
        if len(data.test_dataset.data):
            instruction = data.test_dataset.data[0]["instruction"]
        else:
            instruction = data.train_dataset.data[0]["instruction"]

    elif eval.evaluate_example == "random":
        if len(data.test_dataset.data):
            random_idx = random.randint(0, len(data.test_dataset.data) - 1)
            instruction = data.test_dataset.data[random_idx]["instruction"]
        else:
            random_idx = random.randint(0, len(data.train_dataset.data) - 1)
            instruction = data.train_dataset.data[random_idx]["instruction"]

    elif isinstance(eval.evaluate_example, int):
        index = eval.evaluate_example
        if len(data.test_dataset.data) > index:
            instruction = data.test_dataset.data[index]["instruction"]
        elif len(data.train_dataset.data) > index:
            instruction = data.train_dataset.data[index]["instruction"]
        else:
            raise IndexError(f"Index {index} is out of range for both test and training datasets.")

    else:
        raise ValueError(f"Unknown evaluation example type: {eval.evaluate_example}")
    return instruction
