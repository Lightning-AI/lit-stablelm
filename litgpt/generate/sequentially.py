# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import itertools
import logging
import re
import sys
import time
import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional
import warnings

import lightning as L
from lightning_utilities.core.imports import RequirementCache
import torch
from lightning.fabric.accelerators import CUDAAccelerator
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.utilities.init import _materialize_meta_tensors
from typing_extensions import Type
from tqdm import tqdm

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
import litgpt.generate.base as generate_base
from litgpt.model import Block, build_mask_cache
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.utils import (
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
    get_default_supported_precision
)


@torch.inference_mode()
def sequential(
    model: GPT,
    root: torch.device,
    max_seq_length: int,
    devices: int
):
    if model.config.n_layer < devices:
        raise ValueError(
            f"The number of layers in the model must be larger than the number of devices, but got"
            f" n_layer={model.config.n_layer} and devices={devices}."
        )

    # The last device might get fewer layers if number of layers not evenly divisible by device count
    max_layers_per_device = math.ceil(model.config.n_layer / devices)
    # dictates where each block should be instantiated
    mapping = layer_to_device(model, chunk_on=Block, chunk_size=max_layers_per_device)

    if set(mapping.values()) != set(range(devices)):
        # TODO: support smarter partitioning schemes
        raise RuntimeError(
            f"Not able to distribute the {model.config.n_layer} layers across {devices} devices."
            " Try running with a lower number of devices."
        )

    num_layers_per_device = {i: sum(1 for v in mapping.values() if v == i) for i in range(devices)}

    # materialize each block on the appropriate device
    with tqdm(total=len(mapping), desc="Moving submodules") as pbar:
        for path, target_index in mapping.items():
            submodule = model.get_submodule(path)
            target_device = torch.device(root.type, target_index)

            pbar.set_description(f"Moving {path!r} to {target_device}")
            pbar.update(1)

            # submodules loaded by the checkpoint will be on CPU (if no quantization). move them
            replace_device(submodule, replace=torch.device("cpu"), by=target_device)
            # in case the checkpoint was partial, materialize leftover metas
            _materialize_meta_tensors(submodule, target_device)
    # rebuild odd ends
    with root:
        # Setting `max_seq_length` forces other members to be built
        if model.max_seq_length == max_seq_length:
            model.max_seq_length = max_seq_length + 1
        model.max_seq_length = max_seq_length
    # and everything that is not a block in the root
    _materialize_meta_tensors(model, root)
    replace_device(model, replace=torch.device("cpu"), by=root)

    if devices > 1:
        # install hooks to move layer inputs/output between devices
        for layer_num, (path, target_index) in enumerate(mapping.items()):
            submodule = model.get_submodule(path)
            if layer_num >= num_layers_per_device[target_index]:
                # we need to move the block input on the boundaries between devices
                # and also on every non-root device because the RoPE and mask cache is shared
                # TODO: the second case could be optimized and then we would only need this hook for
                # `layer_num in [layers_per_rank * i - 1 for i in range(1, devices + 1)]`
                target_device = torch.device(root.type, target_index)
                submodule.register_forward_pre_hook(partial(move_block_input, target_device))
            if layer_num == model.config.n_layer - 1:
                submodule.register_forward_hook(partial(move_block_output, root))

    return model


def layer_to_device(
    module: torch.nn.Module, chunk_on: Type[torch.nn.Module], chunk_size: int
) -> "OrderedDict[str, int]":
    """Create a mapping from layer (block) to device."""
    # this assumes that the definition order is the same as the execution order
    hits = [name for name, submodule in module.named_modules() if isinstance(submodule, chunk_on)]
    return OrderedDict((name, i // chunk_size) for i, name in enumerate(hits))


def move_block_input(device: torch.device, module: torch.nn.Module, ins):
    """``forward_pre_hook`` to move a Block's input before forward."""
    # during inference, none of the inputs are None: x, cos, sin, mask, input_pos
    return tuple(t.to(device) for t in ins)


def move_block_output(device: torch.device, module: torch.nn.Module, ins, outs) -> torch.Tensor:
    """``forward_hook`` to move a Block's output after forward."""
    return outs.to(device)


def replace_device(module: torch.nn.Module, replace: torch.device, by: torch.device) -> torch.nn.Module:
    for name, submodule in module.named_modules():
        tensors = dict(
            itertools.chain(submodule.named_parameters(recurse=False), submodule.named_buffers(recurse=False))
        )
        if not tensors:
            continue
        devices = {t.device for t in tensors.values()}
        if len(devices) != 1:
            # since this is using `submodule.to`, different devices in the same submodule is a problem
            path_to_device = {f"{name}.{p}": t.device for p, t in tensors.items()}
            raise ValueError(f"Found multiple devices: {path_to_device}")
        if devices.pop() == replace:
            submodule.to(by)
    return module


@torch.inference_mode()
def main(
    checkpoint_dir: Path,
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    prompt_chunksize: int = 1,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Generation script that partitions layers across devices to be run sequentially.

    Generates text samples based on a pre-trained model and tokenizer.

    Args:
        checkpoint_dir: The checkpoint directory to load.
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        prompt_chunksize: If even the shortest prompt is longer than the KV
            cache, prompts are processed in chunks of this size in the
            prefill phase. Once the shortest has been processed to the
            end, we proceed with chunk size 1.
            Defaults to 1, but larger values are recommended for long prompts.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if compile:
            raise NotImplementedError  # untested
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. "
                "This may result in errors when using quantization."
            )
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        logging.getLogger("lightning.fabric.plugins.precision.bitsandbytes").setLevel(logging.DEBUG)
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cuda", plugins=plugins)

    total_devices = CUDAAccelerator.auto_device_count()
    print(f"Using {total_devices} devices", file=sys.stderr)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    )
    prompt = prompt_style.apply(prompt)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    # cannot use `init_module` because if bitsandbytes is used, the Linear layers will be replaced
    # which means that the weights will get quantized on cuda:0 on checkpoint load. we need to load and then convert
    # still, use init_tensor for the precision
    with fabric.init_tensor(), torch.device("meta"):
        model = GPT(config)
    print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    state_dict = torch.load(str(checkpoint_path), mmap=True, map_location="cpu")
    # TODO: this assumes that the model fits on CPU. Use lazy_load and make the materialization checkpoint aware
    model.load_state_dict(state_dict, assign=True)
    print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model = fabric.setup_module(model, move_to_device=False)

    t0 = time.perf_counter()
    model = sequential(model, fabric.device, max_returned_tokens, total_devices)
    print(f"Time to sequential-ize the model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    if compile:
        # TODO: raises an internal compile AssertionError caused by fabric.strategy.precision.forward_context
        raise NotImplementedError
        # silence developer warning on nightly builds
        # https://github.com/pytorch/pytorch/blob/v2.2.0-rc5/torch/_inductor/ir.py#L4166
        pattern = re.compile(".*DeviceCopy in input program.*")
        logging.getLogger("torch._inductor.utils").addFilter(lambda record: not pattern.search(record.getMessage()))
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        # cannot use cudagraphs because it doesn't support multiple device indices
        # https://github.com/pytorch/pytorch/blob/v2.2.0-rc5/torch/_inductor/compile_fx.py#L371-L375

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate_base.generate(
            model=model,
            prompts=[encoded],
            max_returned_tokens=max_returned_tokens,
            prompt_chunksize=prompt_chunksize,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=tokenizer.eos_id,
        )[0]
        t = time.perf_counter() - t0
        model.clear_kv_cache()
        print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
