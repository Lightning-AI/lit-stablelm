# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#
# This file implements the LitGPT Python API
from pathlib import Path
import sys
import time
from typing import Any, Callable, List, Literal, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm
import torch
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.accelerators import CUDAAccelerator

from litgpt.model import GPT
from litgpt.config import name_to_config, Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.sequentially import sequential
from litgpt.generate.tp import tensor_parallel
from litgpt.generate.base import generate as generate_fn
from litgpt.chat.base import generate as stream_generate_fn
from litgpt.prompts import (
    load_prompt_style,
    has_prompt_style,
    save_prompt_style,
    PromptStyle
)
from litgpt.utils import (
    auto_download_checkpoint,
    check_file_size_on_cpu_and_warn,
    check_nvlink_connectivity,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
    save_config,
)


class LLM(torch.nn.Module):
    def __init__(
        self,
        model: GPT,
        preprocessor=None,
        prompt_style: PromptStyle = None,
        devices: Union[int, List[int]] = None,
        config: Config = None,
        checkpoint_dir: Path = None,
        fabric: L.Fabric = None,
        generate_strategy: Optional[Literal["sequential", "tensor_parallel"]] = None,
        kv_cache_initialized: bool = False,
        fixed_kv_cache_size: Union[int, Literal["max_model_supported"], None] = None
    ) -> None:
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.devices = devices
        self.prompt_style = prompt_style
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.fabric = fabric
        self.generate_strategy = generate_strategy
        self.kv_cache_initialized = kv_cache_initialized
        self.fixed_kv_cache_size = fixed_kv_cache_size
        self.prev_generated_seq_length = 0

    """
    LLM model class for inference, pretraining, and finetuning.

    Example:
        from litgpt.api import LLM

        llm = LLM.load("microsoft/phi-2")
        text = llm.generate("What do Llamas eat?", top_k=1)
        print(text)
    """
    @property
    def tokenizer(self):
        return self.preprocessor.tokenizer

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits = self.model(input_ids)
        if target_ids is not None:
            if loss_fn is None:
                loss_fn = chunked_cross_entropy
            loss = loss_fn(logits[..., :-1, :], target_ids[..., 1:])
            return logits, loss
        else:
            return logits

    def trainer_setup(self, trainer_ckpt: Optional[Path] = None) -> None:
        """Initializes the model checkpoint for PyTorch Lightning Trainer contexts"""
        self.model = GPT(self.config)

        if trainer_ckpt is not None:
            # strip the object name key from the state_dict
            state_dict = torch.load(trainer_ckpt, weights_only=True)["state_dict"]
            first_key = next(iter(state_dict))
            prefix = first_key.split(".")[0] + "."
            keys_to_modify = [key for key in state_dict if key.startswith(prefix)]
            for key in keys_to_modify:
                new_key = key.replace(prefix, "", 1)
                state_dict[new_key] = state_dict.pop(key)

            self.load_state_dict(state_dict, strict=True)

        elif self.checkpoint_dir is not None:
            state_dict = torch.load(self.checkpoint_dir / "lit_model.pth", weights_only=False)
            self.load_state_dict(state_dict, strict=False)

        else:
            raise ValueError(
                "No checkpoint found. Either provide a valid path via `trainer_ckpt` "
                "or ensure that `self.checkpoint_dir` points to a folder containing a `lit_model.pth` weight file."
            )

    def save(self, out_dir: Optional[Path] = None, prompt_style: Optional[PromptStyle] = None) -> None:
        out_dir = Path(out_dir)
        save_path = out_dir / "lit_model.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if prompt_style is None:
            prompt_style = PromptStyle.from_config(self.config)
        if self.fabric is None:
            torch.save(self.state_dict(), save_path)
        else:
            self.fabric.save(save_path, self.state_dict())

        if self.fabric is None or self.fabric.global_rank == 0:
            # If initialization a model with random weights, the checkpoint dir can be none
            if self.checkpoint_dir is not None:
                copy_config_files(Path(self.checkpoint_dir), save_path.parent)
            else:
                save_config(self.config, out_dir)

            save_prompt_style(prompt_style, save_path.parent)

    @classmethod
    def load(
        cls,
        model: str,
        init: Optional[Literal["pretrained", "random"]] = "pretrained",
        tokenizer_dir: Optional[Path] = None,
        access_token: Optional[str] = None,
        distribute: Optional[Literal["auto"]] = "auto"
    ) -> "LLM":
        """
        Loads the LLM from a local directory or model hub.

        Arguments
            model: A local path to a directory containing the model weights or a valid model name.
               You can get a list of valid model names via the `litgpt download list` command line argument.
            init: If "pretrained" (default), downloads the model from the HF Hub if a local model can't be found at the `model`
                directory name; otherwise loads the model from the local directory.
                If "random", initializes the `model` with random weights.
            tokenizer_dir: An optional tokenizer directory if `model` is not a checkpoint directory, or if a user
                wants to use a different tokenizer instead.
            access_token: Optional API token to access models with restrictions when using `init="pretrained"`.
            distribute: If "auto" (default), initializes the model on a single GPU if available and otherwise on the CPU.
                To have more control over the model distribution strategy and utilize multiple GPUs, you can set
                `llm = LLM.load(..., distribute=None)` and call `llm.distribute(...)` manually.
        """

        allowed_init = {"pretrained", "random"}

        if init == "pretrained":
            checkpoint_dir = auto_download_checkpoint(model_name=model, access_token=access_token, ignore_tokenizer_files=tokenizer_dir is not None)
            config = Config.from_file(checkpoint_dir / "model_config.yaml")

        elif init == "random":
            checkpoint_dir = None
            try:
                config = Config.from_name(model)
            except ValueError:
                print(f"Model name {model} is not supported.\n")
                available_models = "\n".join(sorted(name_to_config))
                print(f"Available values:\n{available_models}")
                return

        else:
            raise ValueError(f"Invalid init option: {init}. Must be one of {allowed_init}")

        torch.set_float32_matmul_precision("high")

        if tokenizer_dir is not None:
            tokenizer_dir = extend_checkpoint_dir(Path(tokenizer_dir))
            tokenizer = Tokenizer(tokenizer_dir)
        elif checkpoint_dir is not None:
            tokenizer = Tokenizer(checkpoint_dir)
        else:
            raise ValueError("Provide a path to a tokenizer directory via the `tokenizer_dir` setting.")

        if checkpoint_dir is not None:
            prompt_style = (
                load_prompt_style(checkpoint_dir)
                if has_prompt_style(checkpoint_dir)
                else PromptStyle.from_config(config)
            )
        else:
            prompt_style = PromptStyle.from_config(config)

        if distribute == "auto":
            if torch.cuda.is_available():
                accelerator = "cuda"
            elif torch.backends.mps.is_available():
                accelerator = "mps"
            else:
                accelerator = "cpu"

            fabric = L.Fabric(
                accelerator=accelerator,
                devices=1,
                precision=get_default_supported_precision(training=False),
            )

            with fabric.init_module(empty_init=False):
                model = GPT(config)
            model.eval()
            preprocessor = Preprocessor(tokenizer, device=fabric.device)

            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / "lit_model.pth"
                check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)
                load_checkpoint(fabric, model, checkpoint_path)

            model = fabric.setup_module(model)

        else:
            preprocessor = Preprocessor(tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
            model = None
            fabric = None

        return cls(
            model=model, preprocessor=preprocessor, prompt_style=prompt_style,
            config=config, checkpoint_dir=checkpoint_dir, fabric=fabric, generate_strategy=None,
            kv_cache_initialized=False, fixed_kv_cache_size=False
        )

    def distribute(
        self,
        accelerator: Literal["cpu", "cuda", "auto"] = "auto",
        devices: Union[int, Literal["auto"]] = "auto",
        precision: Optional[Any] = None,
        quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
        generate_strategy: Optional[Literal["sequential", "tensor_parallel"]] = None,
        fixed_kv_cache_size: Union[int, Literal["max_model_supported"], None] = None
    ) -> None:
        """
        Moves the model onto specified devices for single-GPU or multi-GPU inference

        accelerator: Which device type to load the model on ("cpu", "gpu", "mps", "cuda", or "auto")
        devices: The number of devices (1, 2, etc.) or "auto", which uses all available devices
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
            For instance, "32-true", "16-mixed", "16-true", "bf16-mixed", "bf16-true".
            For more details, see https://lightning.ai/docs/fabric/stable/api/fabric_args.html#precision
        generate_strategy: Whether to use a sequential model generation strategy. The "sequential" settings allows running
            models that wouldn't fit in a single card by partitioning the transformer blocks across
            all devices and running them sequentially. Sequential generation may be slower but allows using larger models.
            Note that sequential generation sets `fixed_kv_cache_size="max_model_supported"`. You can set it to a lower integer
            value, `fixed_kv_cache_size=256` to reduce memory. The `fixed_kv_cache_size` value determines the maximum number
            of tokens that can be returned via `llm.generate(...)`.
        fixed_kv_cache_size: If set to an integer value or "max_model_supported" is set, the kv-cache won't be resized dynamically
            during `llm.generate` calls. Use this setting if you plan to compile the model or use `generate_strategy="sequential`.
            Note that the chosen `fixed_kv_cache_size` value determines the maximum number of tokens that can be returned in `llm.generate(...)`.
        """

        if self.checkpoint_dir is None:
            raise NotImplementedError(
                "The LLM was initialized with init='random' but .distribute() "
                "currently only supports pretrained weights."
            )

        allowed_accelerators = {"cpu", "gpu", "cuda", "mps", "auto"}
        if accelerator not in allowed_accelerators:
            raise ValueError(f"Invalid accelerator: {accelerator}. Must be one of {allowed_accelerators}.")

        if accelerator == "auto":
            if torch.cuda.is_available():
                accelerator = "cuda"
            elif torch.backends.mps.is_available():
                accelerator = "mps"
            else:
                accelerator = "cpu"

        if generate_strategy in ("sequential", "tensor_parallel") and accelerator not in ("cuda", "gpu"):
            raise NotImplementedError(f"generate_strategy='{generate_strategy}' is only supported for accelerator='cuda'|'gpu'.")

        if devices == "auto":
            if generate_strategy in ("sequential", "tensor_parallel"):
                total_devices = CUDAAccelerator.auto_device_count()
            else:
                total_devices = 1
        elif isinstance(devices, int):
            use_devices = calculate_number_of_devices(devices)
            total_devices = CUDAAccelerator.auto_device_count()
            if use_devices > total_devices:
                raise ValueError(
                    f"You selected more devices ({use_devices}) than available in your system ({total_devices})."
                )
            else:
                total_devices = use_devices

            if total_devices > 1 and generate_strategy not in ("sequential", "tensor_parallel"):
                raise NotImplementedError(
                    "Support for multiple devices is currently only implemented for generate_strategy='sequential'|'tensor_parallel'."
                )

        else:
            raise ValueError(f"devices argument must be an integer or 'auto', got {devices}")

        print(f"Using {total_devices} device(s)", file=sys.stderr)

        if precision is None:
            precision = get_default_supported_precision(training=False)

        plugins = None
        if quantize is not None and quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("The combination of quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

        # set "ddp" as the strategy for the launching functionality, but there's no data-parallelism
        if generate_strategy != "tensor_parallel":
            fabric = L.Fabric(
                accelerator=accelerator,
                devices=1,  # Otherwise sequential wouldn't work, see litgpt/generate/sequentially.py
                #devices=devices,
                precision=precision,
                plugins=plugins
            )
        else:
            fabric = L.Fabric(
                accelerator=accelerator,
                devices=total_devices,
                strategy="ddp",
                precision=precision,
                plugins=plugins
            )
            if torch.cuda.is_available() and fabric.accelerator.auto_device_count() > 1:
                check_nvlink_connectivity(fabric)
                fabric.launch()

        self.kv_cache_initialized = False
        if generate_strategy is None:
            with fabric.init_module(empty_init=(total_devices > 1)):
                model = GPT(self.config)
            model.eval()

            if self.checkpoint_dir is not None:
                load_checkpoint(fabric, model, self.checkpoint_dir / "lit_model.pth")

            model = fabric.setup_module(model)

            if fixed_kv_cache_size is not None:
                if fixed_kv_cache_size is None or fixed_kv_cache_size == "max_model_supported":
                    kv_cache_size = model.max_seq_length
                else:
                    kv_cache_size = fixed_kv_cache_size
                model.set_kv_cache(batch_size=1, max_seq_length=kv_cache_size, device=fabric.device)
                self.kv_cache_initialized = True
                self.fixed_kv_cache_size = fixed_kv_cache_size

        elif generate_strategy in ("sequential", "tensor_parallel"):

            with fabric.init_tensor(), torch.device("meta"):
                model = GPT(self.config)
            model.eval()

            if generate_strategy == "sequential":
                state_dict = torch.load(str(self.checkpoint_dir / "lit_model.pth"), mmap=True, map_location="cpu", weights_only=False)
                model.load_state_dict(state_dict, assign=True)
                model = fabric.setup_module(model, move_to_device=False)

                if fixed_kv_cache_size is None:
                    fixed_kv_cache_size = "max_model_supported"
                if fixed_kv_cache_size == "max_model_supported":
                    kv_cache_size = model.max_seq_length
                else:
                    kv_cache_size = fixed_kv_cache_size

                model = sequential(model, fabric.device, kv_cache_size, total_devices)
                self.fixed_kv_cache_size = fixed_kv_cache_size

            elif generate_strategy == "tensor_parallel":
                if fabric.global_rank == 0:
                    pbar = tqdm(total=fabric.world_size, desc="Loading model weights")
                for rank in range(fabric.world_size):
                    if fabric.global_rank == rank:
                        state_dict = torch.load(str(self.checkpoint_dir / "lit_model.pth"), mmap=True, map_location="cpu", weights_only=False)
                        model.load_state_dict(state_dict, assign=True)

                        # cannot use `.setup_module` because it will wrap with DDP
                        model = fabric._precision.convert_module(model)
                        model = tensor_parallel(fabric, model)

                        with fabric.init_tensor():
                            if fixed_kv_cache_size is None:
                                fixed_kv_cache_size = "max_model_supported"
                            if fixed_kv_cache_size == "max_model_supported":
                                kv_cache_size = model.max_seq_length
                            else:
                                kv_cache_size = fixed_kv_cache_size
                            model.max_seq_length = kv_cache_size
                            # the rope cache which is on meta device
                            model.cos, model.sin = model.rope_cache()
                            # enable the kv cache
                            model.set_kv_cache(batch_size=1)
                        model.eval()
                        model = fabric.to_device(model)

                    fabric.barrier()
                    if fabric.global_rank == 0:
                        pbar.update(1)

                if fabric.global_rank == 0:
                    pbar.close()

            self.kv_cache_initialized = True

        else:
            raise ValueError(f"Unsupported generate_strategy: {generate_strategy}")

        self.model = model
        self.fabric = fabric
        self.preprocessor.device = fabric.device

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        prompt_chunksize: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        return_as_token_ids: bool = False,
        stream: bool = False
    ) -> Union[str, torch.Tensor]:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        Arguments:
            model: The model to use.
            prompt: The prompt string to use for generating the samples.
            max_new_tokens: The maximum number of new tokens to return.
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
            return_as_token_ids: If True, returns the token IDs as a torch.Tensor. Otherwise, returns the decoded text as a string.
            stream: If True, returns a generator that yields tokens as they are generated.
                At the moment, this setting is slower and may use more memory than the non-streaming version.
                We plan to resolve this in the future.
        """
        if self.model is None:
            raise AttributeError(
                "The model is not initialized yet; use the .distribute() "
                "or .trainer_setup() method to initialize the model."
            )
        input_ids = self._text_to_token_ids(prompt)
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        if not self.kv_cache_initialized:
            if self.fabric is not None:
                device = self.fabric.device
            else:
                device = self.preprocessor.device
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.kv_cache_initialized = True

        # Dynamically grow the kv cache size if necessary
        self.model.clear_kv_cache()
        if not self.fixed_kv_cache_size and self.prev_generated_seq_length < max_returned_tokens:
            tmp_device = self.model.mask_cache.device
            self.model.clear_kv_cache()
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=tmp_device)

        self.prev_generated_seq_length = max_returned_tokens
        self.model.eval()

        def iterator():
            outputs = stream_generate_fn(
                model=self.model,
                prompt=input_ids,
                max_returned_tokens=max_returned_tokens,
                prompt_chunksize=prompt_chunksize,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=([self.preprocessor.tokenizer.eos_id],),
            )
            if return_as_token_ids:
                yield from outputs
            else:
                for output in outputs:
                    yield self.preprocessor.decode(output)
            return

        if stream:
            outputs = iterator()
        else:
            outputs = generate_fn(
                model=self.model,
                prompt=input_ids,
                max_returned_tokens=max_returned_tokens,
                prompt_chunksize=prompt_chunksize,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=self.preprocessor.tokenizer.eos_id,
                include_prompt=False,
            )

        if stream:
            return outputs
        elif return_as_token_ids:
            return outputs
        else:
            return self.preprocessor.decode(outputs)

    def _text_to_token_ids(self, prompt):
        """Utility method to convert a prompt text to token IDs"""
        prompt = self.prompt_style.apply(prompt)
        input_ids = self.preprocessor.encode(prompt)
        return input_ids

    def benchmark(self, num_iterations=1, **kwargs):
        """
        A wrapper around the .generate() method to calculate runtime performance.

        Arguments:
        num_iterations: How often the `.generate()` call is repeated.
        kwargs: Keyword arguments that are passed to the .generate() method.
        """
        benchmark_dict = {}

        for i in range(num_iterations):

            time_to_first_token = None
            t0 = time.perf_counter()
            outputs = self.generate(**kwargs)

            if kwargs.get("stream", False):
                gen_outputs = []
                for e in outputs:
                    if time_to_first_token is None:
                        t1 = time.perf_counter()
                        time_to_first_token = t1 - t0
                    gen_outputs.append(e)
                outputs = "".join(gen_outputs)
            else:
                outputs = self.generate(**kwargs,)
            benchmark_dict.setdefault("Seconds total", []).append(time.perf_counter() - t0)

            benchmark_dict.setdefault("Seconds to first token", []).append(time_to_first_token)
            tokens_generated = self.preprocessor.encode(outputs).size(0)
            benchmark_dict.setdefault("Tokens generated", []).append(tokens_generated)
            benchmark_dict.setdefault("Inference speed in tokens/sec", []).append(
                benchmark_dict["Tokens generated"][-1] / benchmark_dict["Seconds total"][-1]
            )
            if self.fabric is not None and self.fabric.device.type == "cuda":
                benchmark_dict.setdefault("Total GPU memory allocated in GB", []).append(
                    torch.cuda.max_memory_allocated() / 1e9
                )

        return outputs, benchmark_dict


class Preprocessor:
    """
    Preprocesser class for tokenization and de-tokenization.
    """

    def __init__(self, tokenizer: Tokenizer, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, device=self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)


def calculate_number_of_devices(devices):
    """
    Utility function to calculate the number of devices.
    """
    num_devices = devices if isinstance(devices, int) else len(devices) if isinstance(devices, list) else 0
    return num_devices


def benchmark_dict_to_markdown_table(data):
    """
    Converts .benchmark() outputs to a markdown table
    """
    markdown_table = "| Metric                              | Mean                        | Std Dev                     |\n"
    markdown_table += "|-------------------------------------|-----------------------------|-----------------------------|\n"

    for key, values in data.items():
        mean_value = np.mean(values)
        std_dev_value = np.std(values, ddof=1)

        formatted_mean = f"{mean_value:.2f}"
        formatted_std_dev = f"{std_dev_value:.2f}"

        markdown_table += f"| {key.ljust(35)} | {formatted_mean.ljust(27)} | {formatted_std_dev.ljust(27)} |\n"

    return markdown_table


def pull_request_benchmark_util(model_name="microsoft/phi-2", num_iterations=6):

    def print_table(header, data):
        print(f"\n### {header}\n")
        markdown_table = (
            f"| Metric                               | First Iteration | "
            f"Iter 2-{num_iterations} Mean     | Iter 2-{num_iterations} Standard Dev.  |\n"
            f"|--------------------------------------|-----------------|"
            f"-------------------|-------------------------|\n"
        )

        for key, value in data.items():
            first_iteration = f"{value[0]:.2f}" if value[0] is not None else 'N/A'
            clean_values = [v for v in value[1:] if v is not None]

            if clean_values:
                mean_value = np.mean(clean_values)
                std_dev_value = np.std(clean_values, ddof=1)
                mean_str = f"{mean_value:.2f}"
                std_dev_str = f"{std_dev_value:.2f}"
            else:
                mean_str = "N/A"
                std_dev_str = "N/A"

            markdown_table += (
                f"| {key:<36} | {first_iteration:<15} | "
                f"{mean_str:<17} | {std_dev_str:<23} |\n"
            )
        print(markdown_table)

    import subprocess
    try:
        g_hash = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
        print(f"Git Commit Hash: {g_hash}")
    except subprocess.CalledProcessError:
        print("Git Commit Hash: N/A")
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1st table
    llm = LLM.load(
        model=model_name,
    )
    text, bench_d = llm.benchmark(num_iterations=num_iterations, prompt="What do llamas eat?", top_k=1)
    print_table(f"Defaults ({model_name}), 1st time", bench_d)
    del llm

    # 2nd table
    llm = LLM.load(
        model=model_name,
    )
    text, bench_d = llm.benchmark(num_iterations=num_iterations, prompt="What do llamas eat?", top_k=1)
    print_table(f"Defaults ({model_name}), 2nd time", bench_d)
    del llm

    # 3nd table
    llm = LLM.load(
        model=model_name,
    )
    text, bench_d = llm.benchmark(num_iterations=num_iterations, prompt="What do llamas eat?", top_k=1, stream=True)
    print_table("stream=True", bench_d)
    del llm

    # 4th table
    llm = LLM.load(
        model=model_name,
        distribute=None
    )
    llm.distribute(fixed_kv_cache_size=500)

    text, bench_d = llm.benchmark(num_iterations=num_iterations, prompt="What do llamas eat?", top_k=1, stream=True)
    print_table("stream=True + fixed_kv_cache=500", bench_d)
