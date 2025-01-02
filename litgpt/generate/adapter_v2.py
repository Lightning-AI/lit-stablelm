# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional
import warnings

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning_utilities.core.imports import RequirementCache

from litgpt import PromptStyle, Tokenizer
from litgpt.adapter_v2 import GPT, Config
from litgpt.generate.base import generate
from litgpt.prompts import has_prompt_style, load_prompt_style
from litgpt.utils import (
    check_file_size_on_cpu_and_warn,
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
    get_default_supported_precision,
    lazy_load
)


def main(
    checkpoint_dir: Path,
    prompt: str = "What food do llamas eat?",
    input: str = "",
    adapter_path: Path = Path("out/finetune/adapter-v2/final/lit_model.pth.adapter_v2"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    max_new_tokens: int = 100,
    prompt_chunksize: int = 1,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
    precision: Optional[str] = None,
) -> None:
    """For models finetuned with `litgpt finetune adapter_v2`.

    Generates a response based on a given instruction and an optional input. This script will only work with
    checkpoints from the instruction-tuned adapter v2 model. See ``litgpt.finetune.adapter_v2``.

    Args:
        checkpoint_dir: The path to the checkpoint folder with pretrained model weights.
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            ``litgpt.finetune.adapter_v2``.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
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
        precision: Indicates the Fabric precision setting to use.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. "
                "This may result in errors when using quantization."
            )
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    )

    prompt = prompt_style.apply(prompt, input=input)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    adapter_checkpoint = lazy_load(adapter_path)
    checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model = fabric.setup(model)

    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(
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

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    fabric.print(output)

    tokens_generated = y.size(0) - prompt_length
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
