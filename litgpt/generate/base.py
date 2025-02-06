# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, Optional, Tuple, List, Union, Iterator, Dict
import warnings

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning_utilities.core.imports import RequirementCache

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import has_prompt_style, load_prompt_style, PromptStyle
from litgpt.utils import (
    check_file_size_on_cpu_and_warn,
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint
)


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1)
    res_shape, fin_dim = probs.shape[:-1], probs.shape[-1]
    if probs.ndim > 2:
        probs = probs.view(-1, fin_dim)
    return torch.multinomial(probs, num_samples=1).view(*res_shape)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[..., -1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        logits: Logits of sampling probabilities, (batch_size, num, n_vocab)
        temperature: Sampling temperature, defaults to 1
        top_k: Parameter for top-k sampling, defaults to None
        top_p: Parameter for top-p sampling, defaults to 1

    Returns:
        Token indices, (batch_size, num)
    """
    if not (0.0 <= top_p <= 1.0):
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)  # (batch_size, 1, n_vocab)
    elif logits.ndim != 3:
        raise ValueError(f"logits must be 3D tensor, got {logits.shape}")
    # Now: logits has shape (batch_size, num, n_vocab)
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    else:
        return torch.argmax(logits, dim=-1)


class BatchSampler:
    """
    Performs token sampling given logits in the batch case, in the presence
    of prompts of different length.

    """
    def __init__(
        self,
        prompts: List[torch.Tensor],
        next_token_pos: int,
        sample_kwargs: List[Dict[str, Any]],
    ):
        self.prompts = prompts
        self.next_token_pos = next_token_pos
        self.sample_kwargs = sample_kwargs
        self.batch_size = len(prompts)
        assert len(sample_kwargs) == self.batch_size
        self.prompt_length = [p.shape[0] for p in prompts]
        self.sampling_done = [False] * self.batch_size

    def __call__(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given `logits`, samples tokens for each batch dimension. Tokens are
        taken from prompts as long as they have not been completely processed.
        The mask `mask` contains `True` for batch dimensions where a token is
        sampled, `False` if the token is taken from the prompt or sampling has
        stopped for this dimension.

        Args:
            logits: Logits of shape `(batch_size, num, n_vocab)`

        Returns:
            `(tokens, mask)`, with shape `(batch_size, num)`

        """
        # We only use rows in `logits` corresponding to active dimensions,
        # where a token has to be sampled
        assert logits.ndim == 3
        bs, num, _ = logits.shape
        assert bs == self.batch_size
        assert num > 0
        prompt = self.prompts[0]
        shape = (self.batch_size, num)
        mask = torch.zeros(
            shape, dtype=torch.bool, device=prompt.device
        )
        # Used for dims where sampling has stopped:
        dummy_token = prompt[-1].item()
        tokens = torch.full(
            shape,
            dummy_token,
            dtype=prompt.dtype,
            device=prompt.device
        )
        for i, stopped in enumerate(self.sampling_done):
            if not stopped:
                np = self.next_token_pos
                plen = self.prompt_length[i]
                if np + num > plen:
                    tokens[i] = sample(
                        logits[i], **self.sample_kwargs[i]
                    ).to(dtype=prompt.dtype)
                    mask[i] = True
                if plen > np:
                    end = min(plen - np, num)
                    tokens[i, :end] = self.prompts[i][np:(np + end)].to(dtype=prompt.dtype)
                    mask[i, :end] = False
        self.next_token_pos += num
        return tokens, mask

    def stop_sampling(self, dims: List[int]):
        for dim in dims:
            self.sampling_done[dim] = True

    def append_tokens(
        self,
        token_lists: List[List[int]],
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ):
        bs, num = tokens.shape
        assert len(token_lists) == bs == self.batch_size
        assert mask.shape == (bs, num)
        for b, token_list in enumerate(token_lists):
            for i in range(num):
                if mask[b][i]:
                    token_list.append(tokens[b][i].item())

    def all_dimensions_done(self) -> bool:
        return all(self.sampling_done)


@torch.inference_mode()
def batched_generate_fn(
    model: GPT,
    prompts: List[torch.Tensor],
    max_returned_tokens: int,
    prompt_chunksize: int,
    *,
    sample_args: Union[List[dict], dict],
    stop_tokens: Tuple[List[int], ...] = (),
    include_prompt: bool,
    include_eos: bool,
) -> Iterator[List[Optional[torch.Tensor]]]:
    """
    Generates tokens for a list of prompts, which need not have the same
    length.

    Args:
        model: The model to use.
        prompts: A list of size batch_size, containing 1D tensors.
        max_returned_tokens: The maximum number of tokens to return, including
            the prompt tokens.
        prompt_chunksize: If even the shortest prompt is longer than the KV
            cache, prompts are processed in chunks of this size in the
            prefill phase. Once the shortest has been processed to the
            end, we proceed with chunk size 1.
        sample_args: The dictionary of kwargs to pass to sample() for each
            token and each index in the batch.
        stop_tokens: A tuple of stop sequences. If any of the sequences are
            generated, the generation stops early before max_returned_tokens.
        include_prompt: Whether to output the prompt tokens.
        include_eos: Whether to output the stop tokens if generation stops early.

    Yields:
        A list of tokens for each prompt in the batch, or None if a stop
        sequence has already been encountered for that index in the batch,
        or if the prompt is still being processed for that index (only if
        include_prompt is False). The number of tokens yielded can depend
        on the batch index, can be more than one.
    """
    batch_size = len(prompts)
    assert batch_size > 0, "No prompts are given"
    assert prompt_chunksize > 0, "prompt_chunksize must be positive"
    prompt_size = []
    device = prompts[0].device
    prompt_dtype = prompts[0].dtype
    for prompt in prompts:
        sz = prompt.shape[0]
        assert prompt.ndim == 1 and sz > 0, "Each prompts must be non-empty 1D tensor"
        assert prompt.device == device and prompt.dtype == prompt_dtype, "Prompts must have the same device, dtype"
        prompt_size.append(sz)
    max_prompt_size = max(prompt_size)

    if isinstance(sample_args, dict):
        sample_args = [sample_args] * len(prompts)
    else:
        assert len(sample_args) == batch_size, "sample_args must have the same length as the batch size."

    assert max_returned_tokens > max_prompt_size, f"Not enough space for {max_prompt_size} prompt tokens in a context length of {max_returned_tokens}."
    if model.max_seq_length < max_returned_tokens - 1:
        raise ValueError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    if include_prompt:
        # Return prompts
        yield prompts

    # Prefill phase
    # We prefill up to the minimum of prompt lengths. if this is longer than
    # the max prefill length of the KV cache, we call the model to produce
    # logits in chunks of `prompt_chunksize`. These logits are not used (except
    # the very last one), but the KV cache is served. The prefill phase ends
    # with the last token of the shortest prompt being processed.
    min_prompt_size = min(prompt_size)
    max_prefill_length = model.kv_cache_max_prefill_length()
    if max_prefill_length is None:
        max_prefill_length = min_prompt_size
    token_pos = min([min_prompt_size, max_prefill_length])
    start = 0
    while True:
        inputs = torch.cat(
            [prompt[start:token_pos].view(1, -1) for prompt in prompts],
            dim=0,
        )
        # We may need the last time slice of `all_logits` below:
        for_prefill = start == 0  # First iteration
        all_logits = model(inputs, for_prefill=for_prefill)
        if for_prefill:
            max_tokens_forward = model.kv_cache_max_tokens_forward()
            if prompt_chunksize > max_tokens_forward:
                print(
                    f"prompt_chunksize = {prompt_chunksize} > {max_tokens_forward} = max_tokens_forward. Lowering it to the latter.")
                prompt_chunksize = max_tokens_forward
        start = token_pos
        if token_pos == min_prompt_size:
            break
        chunksize = min(prompt_chunksize, min_prompt_size - token_pos)
        token_pos += chunksize

    # Generation loop: One token per iteration
    assert token_pos == min_prompt_size  # Sanity check
    sampler = BatchSampler(
        prompts=prompts,
        next_token_pos=token_pos,
        sample_kwargs=sample_args,
    )
    stop_progresses = [[0] * len(stop_tokens) for _ in range(batch_size)] # [batch_size, ~len(stop_tokens)]
    yielded_idx = 0
    token_lists = [[] for _ in range(batch_size)]
    input_pos = token_pos - 1
    tokens = None  # First iteration uses `all_logits`
    for _ in range(max_returned_tokens - token_pos):
        # Generate the next token for each prompt in the batch
        if all_logits is not None:
            # Use logits from prefill for the first token after that (last
            # time slice)
            logits = all_logits[:, -1, :].unsqueeze(1)
            all_logits = None
        else:
            logits = model(tokens, input_pos=input_pos)
        tokens, mask = sampler(logits)
        sampler.append_tokens(token_lists, tokens, mask)
        logits = None
        int_tokens = [token.item() for token in tokens]
        mask = mask.squeeze(-1).tolist()

        # Check for stop sequences
        # For each stop sequence, we keep a running total of how many are matched in stop_progress.
        # If the current token matches the next token in the stop sequence, we increment the
        # running total and hold off on yielding the token.
        for batch_idx, (int_token, active, stop_progress) in enumerate(
            zip(int_tokens, mask, stop_progresses)
        ):
            if active:
                for seq_idx, seq in enumerate(stop_tokens):
                    seq_pos = stop_progress[seq_idx]
                    if seq_pos >= len(seq):
                        continue
                    if int_token == seq[seq_pos]:
                        stop_progress[seq_idx] += 1
                        if stop_progress[seq_idx] == len(seq):
                            # Stop sampling for this batch dimension
                            sampler.stop_sampling([batch_idx])
                            break  # Leave inner loop
                    else:
                        stop_progress[seq_idx] = 0

        # Yield tokens that are not part of a stop sequence in progress.
        # If there are no stop sequences, then that's all of them.
        if stop_tokens:
            safe_idx = min(
                len(l) - max(s) for l, s in zip(token_lists, stop_progresses)
            )
        else:
            safe_idx = min(len(l) for l in token_lists)
        if yielded_idx < safe_idx:
            y_tokens = [
                torch.tensor(
                    token_list[yielded_idx:safe_idx],
                    dtype=prompt_dtype,
                    device=device,
                )
                for token_list in token_lists
            ]
            yield y_tokens
            yielded_idx = safe_idx

        # Update input_pos for the next iteration
        input_pos += 1
        # Leave loop if no more active dimensions
        if sampler.all_dimensions_done():
            break

    # Yield remaining tokens (if any)
    safe_idxes = [len(l) for l in token_lists]
    if not include_eos and stop_tokens:
        for i, stop_progress in enumerate(stop_progresses):
            for seq, stop_prog in zip(stop_tokens, stop_progress):
                len_seq = len(seq)
                if stop_prog == len_seq:
                    # Sequence was stopped: Strip off stop sequence
                    safe_idxes[i] -= len_seq
                    break  # Leave inner loop
    max_token_lists = max(safe_idxes)
    if yielded_idx < max_token_lists:
        y_tokens = [
            None if yielded_idx == safe_idx else
            torch.tensor(
                tokens[yielded_idx:safe_idx],
                dtype=prompt_dtype,
                device=device,
            )
            for safe_idx, tokens in zip(safe_idxes, token_lists)
        ]
        yield y_tokens
    return


@torch.inference_mode()
def generate(
    model: GPT,
    prompts: List[torch.Tensor],
    max_returned_tokens: int,
    *,
    prompt_chunksize: int = 1,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    include_prompt: bool = True,
    include_eos: bool = True,
) -> List[torch.Tensor]:
    """
    Takes a list of prompts as input (1D tensors, can be of different lengths)
    and generates tokens as specified.

    Choice of `pro mpt_chunksize`:
    This parameter can speed up inference for long prompts. Let
    `M = min_prompt_size - max_prefill_length`, the minimum prompt length
    minus the max prefill length of the KV cache. If `M > 0`, the prompt
    prefill uses `ceil(M / prompt_chunksize)` sequential steps, calling
    model inference for chunks of size `prompt_chunksize`. For larger
    `prompt_chunksize`, this is faster due to less sequential steps.
    However, KV cache eviction is done in a more coarse-grained manner,
    which can lead to worse performance.

    Args:
        model: The model to use.
        prompts: List of batch_size 1D tensors, each being a prompt sequence
        max_returned_tokens: The maximum number of tokens to return (given plus
            generated).
        prompt_chunksize: If even the shortest prompt is longer than the KV
            cache, prompts are processed in chunks of this size in the
            prefill phase. Once the shortest has been processed to the
            end, we proceed with chunk size 1.
            Defaults to 1, but larger values are recommended for long prompts.
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
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
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying
            the prompt style) to the output.
        include_eos: If true (default), include <eos> token at the end of
            sequences. Otherwise, this token is stipped off.

    """
    token_list = [[] for _ in range(len(prompts))]
    for part in batched_generate_fn(
        model=model,
        prompts=prompts,
        max_returned_tokens=max_returned_tokens,
        prompt_chunksize=prompt_chunksize,
        sample_args=dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ),
        stop_tokens=(([eos_id],) if eos_id is not None else ()),
        include_prompt=include_prompt,
        include_eos=include_eos,
    ):
        for tl, p in zip(token_list, part):
            if p is not None:
                tl.append(p)

    return [torch.cat(parts) for parts in token_list]


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
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Default generation option.

    Generates text samples based on a pre-trained model and tokenizer.

    Note that this is using default dense KV caches, which may require a lot
    of memory.

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
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
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

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    )

    prompt = prompt_style.apply(prompt)
    encoded = tokenizer.encode(prompt).to(device=fabric.device)
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
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True

    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(
            model=model,
            prompts=[encoded],
            max_returned_tokens=max_returned_tokens,
            prompt_chunksize=prompt_chunksize,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=int(tokenizer.eos_id),
        )[0]
        t = time.perf_counter() - t0
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    model.clear_kv_cache()
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
