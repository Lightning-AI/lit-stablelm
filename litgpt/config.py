# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
import yaml
from typing_extensions import Self
from litgpt.utils import find_multiple


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    # General size parameters
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    # Transformer block (structure, normalizations)
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_qk: bool = False
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    parallel_residual: bool = True
    shared_attention_norm: bool = False
    # Transformer block (self-attention)
    n_head: int = 32
    head_size: Optional[int] = None
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    attn_bias: bool = False
    attention_scores_scalar: Optional[int] = None
    sliding_window_size: Optional[int] = None
    sliding_window_layer_placing: Optional[Literal["all", "interleaved"]] = None
    # if `attention_logit_softcapping` is used, cannot use optimized
    # `torch.nn.functional.scaled_dot_product_attention` (which implements
    # Flash attention), may result in higher memory and runtime footprint.
    attention_logit_softcapping: Optional[float] = None
    # Rotary position embedding (RoPE)
    rope_base: int = 10000
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_adjustments: Optional[dict] = None
    # Transformer block (MLP)
    intermediate_size: Optional[int] = None
    bias: bool = True
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    n_expert: int = 0
    n_expert_per_token: int = 0
    # GPT before/after blocks
    scale_embeddings: bool = False
    lm_head_bias: bool = False
    final_logit_softcapping: Optional[float] = None
    input_norm: bool = True

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(f"The config {self.name!r}, needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None:
            self.sliding_window_layer_stride = (
                1 if (self.sliding_window_layer_placing is None or self.sliding_window_layer_placing == "all") else 2
            )

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Optional[Self]:
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(
                    config
                    for config in configs
                    if name == config["hf_config"]["name"]
                    or config["hf_config"]["org"] + "/" + config["hf_config"]["name"] == name
                )
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> Self:
        """Automatically load `model_config.yaml` and if it doesn't exist - a matching config from `litgpt/config.py`."""
        if (config_path := path / "model_config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'model_config.yaml' nor matching config exists.")

    @property
    def mlp_class(self) -> Type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        import litgpt.model
        return getattr(litgpt.model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        # `self.norm_class_name` cannot be the type to keep the config serializable

        from functools import partial

        if self.norm_class_name == "RMSNorm":
            
            from litgpt.model import RMSNorm

            return partial(RMSNorm, add_unit_offset="Gemma" in self.name)

        if self.norm_class_name == "LayerNorm" and "OLMo" in self.name:
            # this makes it equivalent to `torch.nn.functional.layer_norm`
            # that is used by OLMo
            # Table 5 caption in the OLMo paper shows this - https://aclanthology.org/2024.acl-long.841
            return partial(torch.nn.LayerNorm, elementwise_affine=False)

        return getattr(torch.nn, self.norm_class_name)


########################
# Stability AI StableLM
########################
configs = [
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    dict(name="stablelm-base-alpha-3b", hf_config=dict(org="stabilityai", name="stablelm-base-alpha-3b")),
    # https://huggingface.co/stabilityai/stablelm-base-alpha-7b/blob/main/config.json
    dict(
        name="stablelm-base-alpha-7b",
        hf_config=dict(org="stabilityai", name="stablelm-base-alpha-7b"),
        n_head=48,
        n_embd=6144,
        padding_multiple=256,
    ),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/blob/main/config.json
    dict(name="stablelm-tuned-alpha-3b", hf_config=dict(org="stabilityai", name="stablelm-tuned-alpha-3b"), n_head=32),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b/blob/main/config.json
    dict(
        name="stablelm-tuned-alpha-7b",
        hf_config=dict(org="stabilityai", name="stablelm-tuned-alpha-7b"),
        n_head=48,
        n_embd=6144,
        padding_multiple=256,
    ),
    # https://huggingface.co/stabilityai/stablelm-3b-4e1t/blob/main/config.json
    dict(
        name="stablelm-3b-4e1t",
        hf_config=dict(org="stabilityai", name="stablelm-3b-4e1t"),
        padded_vocab_size=50304,
        n_layer=32,
        n_head=32,
        n_embd=2560,
        parallel_residual=False,
        bias=False,
        mlp_class_name="LLaMAMLP",
        intermediate_size=6912,
    ),
    # https://huggingface.co/stabilityai/stablelm-zephyr-3b/blob/main/config.json
    dict(
        name="stablelm-zephyr-3b",
        hf_config=dict(org="stabilityai", name="stablelm-zephyr-3b"),
        padded_vocab_size=50304,
        n_layer=32,
        n_head=32,
        n_embd=2560,
        parallel_residual=False,
        bias=False,
        mlp_class_name="LLaMAMLP",
        intermediate_size=6912,
    ),
]


##########################
# Stability AI StableCode
##########################
stablecode = [
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b/blob/main/config.json
    dict(
        name="stablecode-completion-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablecode-completion-alpha-3b"),
        block_size=16384,
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k/blob/main/config.json
    dict(
        name="stablecode-completion-alpha-3b-4k",
        hf_config=dict(org="stabilityai", name="stablecode-completion-alpha-3b-4k"),
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/blob/main/config.json
    dict(
        name="stablecode-instruct-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablecode-instruct-alpha-3b"),
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stable-code-3b/blob/main/config.json
    dict(
        name="stable-code-3b",
        hf_config=dict(org="stabilityai", name="stable-code-3b"),
        padded_vocab_size=50304,
        n_layer=32,
        n_embd=2560,
        block_size=16384,
        parallel_residual=False,
        bias=False,
        mlp_class_name="LLaMAMLP",
        intermediate_size=6912,
    ),
]
configs.extend(stablecode)


####################
# EleutherAI Pythia
####################
pythia = [
    # https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json
    dict(
        name="pythia-14m",
        hf_config=dict(org="EleutherAI", name="pythia-14m"),
        block_size=512,
        n_layer=6,
        n_embd=128,
        n_head=4,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-31m/blob/main/config.json
    dict(
        name="pythia-31m",
        hf_config=dict(org="EleutherAI", name="pythia-31m"),
        block_size=1024,
        n_layer=6,
        n_embd=256,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    dict(
        name="pythia-70m",
        hf_config=dict(org="EleutherAI", name="pythia-70m"),
        block_size=2048,
        n_layer=6,
        n_embd=512,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    dict(
        name="pythia-160m",
        hf_config=dict(org="EleutherAI", name="pythia-160m"),
        block_size=2048,
        n_layer=12,
        n_embd=768,
        n_head=12,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    dict(
        name="pythia-410m",
        hf_config=dict(org="EleutherAI", name="pythia-410m"),
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    dict(
        name="pythia-1b",
        hf_config=dict(org="EleutherAI", name="pythia-1b"),
        block_size=2048,
        n_embd=2048,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    dict(
        name="pythia-1.4b",
        hf_config=dict(org="EleutherAI", name="pythia-1.4b"),
        block_size=2048,
        n_layer=24,
        n_embd=2048,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    dict(
        name="pythia-2.8b",
        hf_config=dict(org="EleutherAI", name="pythia-2.8b"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    dict(
        name="pythia-6.9b",
        hf_config=dict(org="EleutherAI", name="pythia-6.9b"),
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
    ),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    dict(
        name="pythia-12b",
        hf_config=dict(org="EleutherAI", name="pythia-12b"),
        block_size=2048,
        n_layer=36,
        n_embd=5120,
        n_head=40,
    ),
]
configs.extend(pythia)
for c in pythia:
    # "pythia-14m" and "pythia-31m" don't have deduped version
    if c["name"] in ("pythia-14m", "pythia-31m"):
        continue
    copy = deepcopy(c)
    copy["name"] = f"{c['name']}-deduped"
    copy["hf_config"]["name"] = f"{c['hf_config']['name']}-deduped"
    configs.append(copy)


#################
# TII UAE Falcon
#################
falcon = [
    # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    dict(
        name="falcon-7b{}",
        hf_config=dict(org="tiiuae", name="falcon-7b{}"),
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=32,
        n_head=71,
        n_embd=4544,
        rotary_percentage=1.0,
        n_query_groups=1,
        bias=False,
        # this is not in the config, but in the original model implementation, only for this config
        shared_attention_norm=True,
    ),
    # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
    dict(
        name="falcon-40b{}",
        hf_config=dict(org="tiiuae", name="falcon-40b{}"),
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=60,
        n_head=128,
        n_embd=8192,
        rotary_percentage=1.0,
        n_query_groups=8,
        bias=False,
    ),
]
for c in falcon:
    for kind in ("", "-instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

# https://huggingface.co/tiiuae/falcon-180b/blob/main/config.json
falcon180b = dict(
    name="falcon-180B{}",
    hf_config=dict(org="tiiuae", name="falcon-180B{}"),
    block_size=2048,
    vocab_size=65024,
    padded_vocab_size=65024,
    n_layer=80,
    n_head=232,
    n_embd=14848,
    rotary_percentage=1.0,
    n_query_groups=8,
    bias=False,
)

for kind in ("", "-chat"):
    copy = deepcopy(falcon180b)
    copy["name"] = falcon180b["name"].format(kind)
    copy["hf_config"]["name"] = falcon180b["hf_config"]["name"].format(kind)
    configs.append(copy)

falcon3 = [
    # https://huggingface.co/tiiuae/Falcon3-1B-Base/blob/main/config.json
    dict(
        name="Falcon3-1B{}",
        hf_config=dict(org="tiiuae", name="Falcon3-1B{}"),
        block_size=4096,
        vocab_size=131072,
        padded_vocab_size=131072,
        n_layer=18,
        n_head=8,
        n_query_groups=4,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        rope_base=1000042,
        norm_eps=1e-6,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
    ),
    # https://huggingface.co/tiiuae/Falcon3-3B-Base/blob/main/config.json
    dict(
        name="Falcon3-3B{}",
        hf_config=dict(org="tiiuae", name="Falcon3-3B{}"),
        block_size=32768,
        vocab_size=131072,
        padded_vocab_size=131072,
        n_layer=22,
        n_head=12,
        n_query_groups=4,
        n_embd=3072,
        rotary_percentage=1.0,
        parallel_residual=False,
        rope_base=1000042,
        norm_eps=1e-6,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=9216,
    ),
    # https://huggingface.co/tiiuae/Falcon3-7B-Base/blob/main/config.json
    dict(
        name="Falcon3-7B{}",
        hf_config=dict(org="tiiuae", name="Falcon3-7B{}"),
        block_size=32768,
        vocab_size=131072,
        padded_vocab_size=131072,
        n_layer=28,
        n_head=12,
        n_query_groups=4,
        n_embd=3072,
        rotary_percentage=1.0,
        parallel_residual=False,
        rope_base=1000042,
        norm_eps=1e-6,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=23040,
    ),
    # https://huggingface.co/tiiuae/Falcon3-10B-Base/blob/main/config.json
    dict(
        name="Falcon3-10B{}",
        hf_config=dict(org="tiiuae", name="Falcon3-10B{}"),
        block_size=32768,
        vocab_size=131072,
        padded_vocab_size=131072,
        n_layer=40,
        n_head=12,
        n_query_groups=4,
        n_embd=3072,
        rotary_percentage=1.0,
        parallel_residual=False,
        rope_base=1000042,
        norm_eps=1e-6,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=23040,
    ),
]
for c in falcon3:
    for kind in ("-Base", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


#############################
# OpenLM Research Open LLaMA
#############################
open_LLaMA = [
    # https://huggingface.co/openlm-research/open_llama_3b/blob/main/config.json
    dict(
        name="open_llama_3b",
        hf_config=dict(org="openlm-research", name="open_llama_3b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=26,
        n_embd=3200,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=8640,
    ),
    # https://huggingface.co/openlm-research/open_llama_7b/blob/main/config.json
    dict(
        name="open_llama_7b",
        hf_config=dict(org="openlm-research", name="open_llama_7b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/openlm-research/open_llama_13b/blob/main/config.json
    dict(
        name="open_llama_13b",
        hf_config=dict(org="openlm-research", name="open_llama_13b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(open_LLaMA)

###############
# Meta LLaMA 2
###############
llama_2 = [
    # https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
    dict(
        name="Llama-2-7b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-7b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
    dict(
        name="Llama-2-13b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-13b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
    dict(
        name="Llama-2-70b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-70b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


###############
# Meta LLaMA 3
###############
llama_3 = [
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
    dict(
        name="Llama-3-8B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3-8B{}"),
        block_size=8192,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        rope_base=500000,
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
    dict(
        name="Llama-3.1-8B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3.1-8B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
    dict(
        name="Llama-3-70B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3-70B{}"),
        block_size=8192,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-70B/blob/main/config.json
    dict(
        name="Llama-3.1-70B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3.1-70B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3.1-405B/blob/main/config.json
    dict(
        name="Llama-3.1-405B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3.1-405B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=126,
        n_head=128,
        n_embd=16384,
        n_query_groups=16,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=53248,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
    dict(
        name="Llama-3.2-1B{}",
        hf_config=dict(org="meta-llama", name="Llama-3.2-1B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=16,
        n_embd=2048,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
        rope_base=500000,
        rope_adjustments=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
    dict(
        name="Llama-3.2-3B{}",
        hf_config=dict(org="meta-llama", name="Llama-3.2-3B{}"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=28,
        n_embd=3072,
        n_head=24,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
        rope_base=500000,
        rope_adjustments=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json
    dict(
        name="Llama-3.3-70B-Instruct",
        hf_config=dict(org="meta-llama", name="Llama-3.3-70B-Instruct"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
]
for c in llama_3:
    if c["name"] == "Llama-3.3-70B-Instruct":
        configs.append(c)
        continue
    for kind in ("", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

#########################
# NVIDIA Llama Nemotron
#########################
configs.append(
    dict(
        name="Llama-3.1-Nemotron-70B-Instruct-HF",
        hf_config=dict(org="nvidia", name="Llama-3.1-Nemotron-70B-Instruct-HF"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
)

#################
# Allen AI OLMo
#################
olmo = [
    # https://huggingface.co/allenai/OLMo-1B-hf/blob/main/config.json
    dict(
        name="OLMo-1B-hf",
        hf_config=dict(org="allenai", name="OLMo-1B-hf"),
        vocab_size=50280,
        padded_vocab_size=50304,
        block_size=2048,
        n_embd=2048,
        n_layer=16,
        n_head=16,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="LayerNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
    ),
    # https://huggingface.co/allenai/OLMo-7B-hf/blob/main/config.json
    dict(
        name="OLMo-7B-hf",
        hf_config=dict(org="allenai", name="OLMo-7B-hf"),
        vocab_size=50280,
        padded_vocab_size=50304,
        block_size=2048,
        n_layer=32,
        n_head=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="LayerNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/allenai/OLMo-7B-Instruct-hf/blob/main/config.json
    dict(
        name="OLMo-7B-Instruct-hf",
        hf_config=dict(org="allenai", name="OLMo-7B-Instruct-hf"),
        vocab_size=50280,
        padded_vocab_size=50304,
        block_size=2048,
        n_layer=32,
        n_head=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="LayerNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
]

configs.extend(olmo)

olmo2 = [
    # https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json
    dict(
        name="OLMo-2-1124-7B{}",
        hf_config=dict(org="allenai", name="OLMo-2-1124-7B{}"),
        vocab_size=100278,
        padded_vocab_size=100352,
        block_size=4096,
        n_embd=4096,
        n_layer=32,
        n_head=32,
        n_query_groups=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        norm_eps=1e-06,
        intermediate_size=11008,
        rope_base=500000,
        norm_qk=True,
        post_mlp_norm=True,
        input_norm=False,
    ),
    # https://huggingface.co/allenai/OLMo-2-1124-13B/blob/main/config.json
    dict(
        name="OLMo-2-1124-13B{}",
        hf_config=dict(org="allenai", name="OLMo-2-1124-13B{}"),
        vocab_size=100278,
        padded_vocab_size=100352,
        block_size=4096,
        n_embd=5120,
        n_layer=40,
        n_head=40,
        n_query_groups=40,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        norm_eps=1e-06,
        intermediate_size=13824,
        rope_base=500000,
        norm_qk=True,
        post_mlp_norm=True,
        input_norm=False,
    ),
]

for c in olmo2:
    for kind in ("", "-SFT", "-DPO", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

###############
# Google Gemma
###############
gemma = [
    # https://huggingface.co/google/gemma-2b/blob/main/config.json
    dict(
        name="Gemma-2b",
        hf_config=dict(org="google", name="gemma-2b"),
        scale_embeddings=True,
        vocab_size=256000,
        padding_multiple=64,
        n_embd=2048,
        n_layer=18,
        n_head=8,
        n_query_groups=1,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        intermediate_size=16384,
    ),
    # https://huggingface.co/google/gemma-7b/blob/main/config.json
    dict(
        name="Gemma-7b",
        hf_config=dict(org="google", name="gemma-7b"),
        scale_embeddings=True,
        vocab_size=256000,
        padding_multiple=64,
        n_embd=3072,
        n_layer=28,
        n_head=16,
        head_size=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        intermediate_size=24576,
    ),
    # https://huggingface.co/google/gemma-2-2b/blob/main/config.json
    dict(
        name="Gemma-2-2b",
        hf_config=dict(org="google", name="gemma-2-2b"),
        scale_embeddings=True,
        attention_scores_scalar=256,
        vocab_size=256000,
        block_size=8192,
        sliding_window_size=4096,
        # only layer with idx 0, 2, 4, ... have sliding window attention
        sliding_window_layer_placing="interleaved",
        intermediate_size=9216,
        n_embd=2304,
        n_layer=26,
        n_head=8,
        n_query_groups=4,
        head_size=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        post_attention_norm=True,
        post_mlp_norm=True,
        attention_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    ),
    # https://huggingface.co/google/gemma-2-9b/blob/main/config.json
    dict(
        name="Gemma-2-9b",
        hf_config=dict(org="google", name="gemma-2-9b"),
        scale_embeddings=True,
        attention_scores_scalar=256,
        vocab_size=256000,
        block_size=8192,
        sliding_window_size=4096,
        # only layer with idx 0, 2, 4, ... have sliding window attention
        sliding_window_layer_placing="interleaved",
        intermediate_size=14336,
        n_embd=3584,
        n_layer=42,
        n_head=16,
        n_query_groups=8,
        head_size=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        post_attention_norm=True,
        post_mlp_norm=True,
        attention_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    ),
    # https://huggingface.co/google/gemma-2-27b/blob/main/config.json
    dict(
        name="Gemma-2-27b",
        hf_config=dict(org="google", name="gemma-2-27b"),
        scale_embeddings=True,
        # In Gemma 2 27B attention scores are scaled not by `sqrt(head_size)` (11.31),
        # but by `sqrt(n_emb // n_head)` = sqrt(4608 // 32) = 12
        attention_scores_scalar=144,
        vocab_size=256000,
        block_size=8192,
        sliding_window_size=4096,
        # only layer with idx 0, 2, 4, ... have sliding window attention
        sliding_window_layer_placing="interleaved",
        intermediate_size=36864,
        n_embd=4608,
        n_layer=46,
        n_head=32,
        n_query_groups=16,
        head_size=128,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        post_attention_norm=True,
        post_mlp_norm=True,
        attention_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    ),
]
configs.extend(gemma)
for c in gemma:
    copy = deepcopy(c)
    copy["name"] = f"{c['name']}-it"
    copy["hf_config"]["name"] = f"{c['hf_config']['name']}-it"
    configs.append(copy)

##################
# Google CodeGemma
##################
codegemma = [
    # https://huggingface.co/google/codegemma-7b-it/blob/main/config.json
    dict(
        name="CodeGemma-7b-it",
        hf_config=dict(org="google", name="codegemma-7b-it"),
        scale_embeddings=True,
        vocab_size=256000,
        padding_multiple=64,
        n_embd=3072,
        n_layer=28,
        n_head=16,
        head_size=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        intermediate_size=24576,
    ),
]
configs.extend(codegemma)


##########################
# Stability AI FreeWilly2
##########################
freewilly_2 = [
    # https://huggingface.co/stabilityai/FreeWilly2/blob/main/config.json
    dict(
        name="FreeWilly2",
        hf_config=dict(org="stabilityai", name="FreeWilly2"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    )
]
configs.extend(freewilly_2)


##################
# Meta Code Llama
##################
code_llama = [
    # https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-Python-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Instruct-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Instruct-hf"),
        block_size=2048,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Instruct-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-Instruct-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
]
configs.extend(code_llama)


########################
# garage-bAInd Platypus
########################
platypus = [
    # https://huggingface.co/garage-bAInd/Platypus-30B/blob/main/config.json
    dict(
        name="Platypus-30B",
        hf_config=dict(org="garage-bAInd", name="Platypus-30B"),
        block_size=2048,
        padded_vocab_size=32000,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-06,
        mlp_class_name="LLaMAMLP",
        intermediate_size=17920,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-7B/blob/main/config.json
    dict(
        name="Platypus2-7B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-7B"),
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-13B/blob/main/config.json
    dict(
        name="Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-13B"),
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-70B/blob/main/config.json
    dict(
        name="Platypus2-70B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-70B"),
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    ),
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-13B/blob/main/config.json
    dict(
        name="Camel-Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Camel-Platypus2-13B"),
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-70B/blob/main/config.json
    dict(
        name="Camel-Platypus2-70B",
        hf_config=dict(org="garage-bAInd", name="Camel-Platypus2-70B"),
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    ),
    # https://huggingface.co/garage-bAInd/Stable-Platypus2-13B/blob/main/config.json
    dict(
        name="Stable-Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Stable-Platypus2-13B"),
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-70B-instruct/blob/main/config.json
    dict(
        name="Platypus2-70B-instruct",
        hf_config=dict(org="garage-bAInd", name="Platypus2-70B-instruct"),
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    ),
]
configs.extend(platypus)


##################################
# togethercomputer LLaMA-2-7B-32K
##################################
together_llama2_32k = [
    # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/config.json
    dict(
        name="LLaMA-2-7B-32K",
        hf_config=dict(org="togethercomputer", name="LLaMA-2-7B-32K"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=8,
    )
]
configs.extend(together_llama2_32k)


################
# Microsoft Phi
################
phi = [
    # https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
    dict(
        name="phi-1_5",
        hf_config=dict(org="microsoft", name="phi-1_5"),
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2048,
        n_layer=24,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    # https://huggingface.co/microsoft/phi-2/blob/main/config.json
    dict(
        name="phi-2",
        hf_config=dict(org="microsoft", name="phi-2"),
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2560,
        n_layer=32,
        rotary_percentage=0.4,  # 32 / (n_embd / n_head) = 32 / 80
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
    dict(
        name="Phi-3-mini-4k-instruct",
        hf_config=dict(org="microsoft", name="Phi-3-mini-4k-instruct"),
        vocab_size=32000,
        padded_vocab_size=32064,
        block_size=4096,
        n_embd=3072,
        n_layer=32,
        rotary_percentage=1.0,
        bias=False,
        norm_class_name="RMSNorm",
        intermediate_size=8192,
        mlp_class_name="LLaMAMLP",
        parallel_residual=False,
        sliding_window_size=2048,
        sliding_window_layer_placing="all",
    ),
    # https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json
    dict(
        name="Phi-3-mini-128k-instruct",
        hf_config=dict(org="microsoft", name="Phi-3-mini-128k-instruct"),
        vocab_size=32000,
        padded_vocab_size=32064,
        block_size=131072,
        n_embd=3072,
        n_layer=32,
        rotary_percentage=1.0,
        bias=False,
        norm_class_name="RMSNorm",
        intermediate_size=8192,
        mlp_class_name="LLaMAMLP",
        parallel_residual=False,
        sliding_window_size=262145,
        sliding_window_layer_placing="all",
    ),
    # https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/config.json
    dict(
        name="Phi-3.5-mini-instruct",
        hf_config=dict(org="microsoft", name="Phi-3.5-mini-instruct"),
        vocab_size=32000,
        padded_vocab_size=32064,
        block_size=4096,
        n_embd=3072,
        n_layer=32,
        rotary_percentage=1.0,
        bias=False,
        norm_class_name="RMSNorm",
        intermediate_size=8192,
        mlp_class_name="LLaMAMLP",
        parallel_residual=False,
    ),
    # https://huggingface.co/microsoft/phi-4/blob/main/config.json
    dict(
        name="phi-4",
        hf_config=dict(org="microsoft", name="phi-4"),
        vocab_size=100352,
        padded_vocab_size=100352,
        block_size=16384,
        n_embd=5120,
        n_layer=40,
        n_head=40,
        n_query_groups=10,
        rotary_percentage=1.0,
        bias=False,
        norm_class_name="RMSNorm",
        intermediate_size=17920,
        rope_base=250000,
        mlp_class_name="LLaMAMLP",
        parallel_residual=False,
    ),
]
configs.extend(phi)


#############
# Mistral AI
#############

configs.append(
    # https://huggingface.co/mistralai/mathstral-7B-v0.1/blob/main/config.json
    dict(
        name="Mathstral-7B-v0.1",
        hf_config=dict(org="mistralai", name="mathstral-7B-v0.1"),
        padded_vocab_size=32768,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        sliding_window_size=4096,
        sliding_window_layer_placing="all",
    )
)

mistral = [
    # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
    dict(
        name="Mistral-7B-{}v0.1",
        hf_config=dict(org="mistralai", name="Mistral-7B-{}v0.1"),
        padded_vocab_size=32000,
        block_size=4096,  # should be 32768 but sliding window attention is not implemented
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        sliding_window_size=4096,
        sliding_window_layer_placing="all",
    ),
    # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
    dict(
        name="Mixtral-8x7B-{}v0.1",
        hf_config=dict(org="mistralai", name="Mixtral-8x7B-{}v0.1"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMoE",
        intermediate_size=14336,
        rope_base=1000000,
        n_expert=8,
        n_expert_per_token=2,
    ),
    # https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/blob/main/config.json
    dict(
        name="Mixtral-8x22B-{}v0.1",
        hf_config=dict(org="mistralai", name="Mixtral-8x22B-{}v0.1"),
        padded_vocab_size=32768,
        block_size=65536,
        n_layer=56,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMoE",
        intermediate_size=16384,
        n_head=48,
        n_embd=6144,
        rope_base=1000000,
        n_expert=8,
        n_expert_per_token=2,
    ),
]
for c in mistral:
    for kind in ("", "Instruct-"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)
configs.append(
    # https://huggingface.co/unsloth/mistral-7b-v0.2/blob/main/config.json
    dict(
        name="Mistral-7B-v0.2",
        hf_config=dict(org="unsloth", name="Mistral-7B-v0.2"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json
    dict(
        name="Mistral-7B-Instruct-v0.2",
        hf_config=dict(org="mistralai", name="Mistral-7B-Instruct-v0.2"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/main/config.json
    dict(
        name="Mistral-7B-v0.3",
        hf_config=dict(org="mistralai", name="Mistral-7B-v0.3"),
        padded_vocab_size=32768,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json
    dict(
        name="Mistral-7B-Instruct-v0.3",
        hf_config=dict(org="mistralai", name="Mistral-7B-Instruct-v0.3"),
        padded_vocab_size=32768,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-Large-Instruct-2407/blob/main/config.json
    dict(
        name="Mistral-Large-Instruct-2407",
        hf_config=dict(org="mistralai", name="Mistral-Large-Instruct-2407"),
        padded_vocab_size=32768,
        block_size=32768,
        n_layer=88,
        n_head=96,
        n_embd=12288,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-Large-Instruct-2411/blob/main/config.json
    dict(
        name="Mistral-Large-Instruct-2411",
        hf_config=dict(org="mistralai", name="Mistral-Large-Instruct-2411"),
        padded_vocab_size=32768,
        block_size=32768,
        n_layer=88,
        n_head=96,
        n_embd=12288,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    )
)


############
# TinyLlama
############
tiny_llama = [
    dict(
        name="tiny-llama-1.1b{}",
        hf_config=dict(org="TinyLlama", name="TinyLlama-1.1B{}"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama use FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
    )
]
for c in tiny_llama:
    for kind, hf_postfix in (("", "-intermediate-step-1431k-3T"), ("-chat", "-Chat-v1.0")):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(hf_postfix)
        configs.append(copy)


############
# MicroLlama
############
micro_llama = [
    dict(
        name="micro-llama-300M",
        hf_config=dict(org="keeeeenw", name="MicroLlama"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=16,
        n_embd=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama and MicroLlama use FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
    )
]
configs.extend(micro_llama)


##########################
# Trelis Function Calling
##########################
llama_2_function_calling = [
    # https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2/blob/main/config.json
    dict(
        name="Llama-2-7b-chat-hf-function-calling-v2",
        hf_config=dict(org="Trelis", name="Llama-2-7b-chat-hf-function-calling-v2"),
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        norm_eps=1e-6,
        block_size=4096,
        vocab_size=32000,
        n_head=32,
        n_embd=4096,
        rope_base=10000,
    )
]

configs.extend(llama_2_function_calling)

##########
# Qwen2.5
##########
qwen_2_5 = [
    # https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
    dict(
        name="Qwen2.5-0.5B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-0.5B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=151936,
        n_layer=24,
        n_head=14,
        n_embd=896,
        n_query_groups=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=4864,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json
    dict(
        name="Qwen2.5-1.5B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-1.5B{}"),
        block_size=131072,
        vocab_size=151643,
        padded_vocab_size=151936,
        n_layer=28,
        n_head=12,
        n_embd=1536,
        n_query_groups=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8960,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json
    dict(
        name="Qwen2.5-3B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-3B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=151936,
        n_layer=36,
        n_head=16,
        n_embd=2048,
        n_query_groups=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json
    dict(
        name="Qwen2.5-7B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-7B{}"),
        block_size=131072,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=28,
        n_head=28,
        n_embd=3584,
        n_query_groups=4,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=18944,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-14B/blob/main/config.json
    dict(
        name="Qwen2.5-14B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-14B{}"),
        block_size=131072,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=48,
        n_head=40,
        n_embd=5120,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        norm_eps=1e-5,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json
    dict(
        name="Qwen2.5-32B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-32B{}"),
        block_size=131072,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=64,
        n_head=40,
        n_embd=5120,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=27648,
        norm_eps=1e-5,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-72B/blob/main/config.json
    dict(
        name="Qwen2.5-72B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-72B{}"),
        block_size=131072,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=29568,
        norm_eps=1e-5,
        rope_base=1000000
    ),
]

qwen_2_5_coder = [
    # https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B/blob/main/config.json
    dict(
        name="Qwen2.5-Coder-0.5B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Coder-0.5B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=151936,
        n_layer=24,
        n_head=14,
        n_embd=896,
        n_query_groups=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=4864,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/blob/main/config.json
    dict(
        name="Qwen2.5-Coder-1.5B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Coder-1.5B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=151936,
        n_layer=28,
        n_head=12,
        n_embd=1536,
        n_query_groups=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8960,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-Coder-3B/blob/main/config.json
    dict(
        name="Qwen2.5-Coder-3B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Coder-3B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=151936,
        n_layer=36,
        n_head=16,
        n_embd=2048,
        n_query_groups=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-Coder-7B/blob/main/config.json
    dict(
        name="Qwen2.5-Coder-7B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Coder-7B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=28,
        n_head=28,
        n_embd=3584,
        n_query_groups=4,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=18944,
        norm_eps=1e-6,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-Coder-14B/blob/main/config.json
    dict(
        name="Qwen2.5-Coder-14B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Coder-14B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=48,
        n_head=40,
        n_embd=5120,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        norm_eps=1e-5,
        rope_base=1000000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-Coder-32B/blob/main/config.json
    dict(
        name="Qwen2.5-Coder-32B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Coder-32B{}"),
        block_size=32768,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=64,
        n_head=40,
        n_embd=5120,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=27648,
        norm_eps=1e-5,
        rope_base=1000000
    ),
]

qwen_2_5.extend(qwen_2_5_coder)

qwen_2_5_math = [
    # https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/config.json
    dict(
        name="Qwen2.5-Math-1.5B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Math-1.5B{}"),
        block_size=4096,
        vocab_size=151643,
        padded_vocab_size=151936,
        n_layer=28,
        n_head=12,
        n_embd=1536,
        n_query_groups=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8960,
        norm_eps=1e-6,
        rope_base=10000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-Math-7B/blob/main/config.json
    dict(
        name="Qwen2.5-Math-7B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Math-7B{}"),
        block_size=4096,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=28,
        n_head=28,
        n_embd=3584,
        n_query_groups=4,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=18944,
        norm_eps=1e-6,
        rope_base=10000
    ),
    # https://huggingface.co/Qwen/Qwen2.5-Math-72B/blob/main/config.json
    dict(
        name="Qwen2.5-Math-72B{}",
        hf_config=dict(org="Qwen", name="Qwen2.5-Math-72B{}"),
        block_size=4096,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=29568,
        norm_eps=1e-5,
        rope_base=10000
    ),
]

qwen_2_5.extend(qwen_2_5_math)

for c in qwen_2_5:
    for kind in ("", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

qwq = [
    # https://huggingface.co/Qwen/QwQ-32B-Preview/blob/main/config.json
    dict(
        name="QwQ-32B-Preview",
        hf_config=dict(org="Qwen", name="QwQ-32B-Preview"),
        block_size=131072,
        vocab_size=151643,
        padded_vocab_size=152064,
        n_layer=64,
        n_head=40,
        n_embd=5120,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        attn_bias=True,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=27648,
        norm_eps=1e-5,
        rope_base=1000000
    ),
]

configs.extend(qwq)


#############    
# Salamandra
#############
salamandra = [
    # https://huggingface.co/BSC-LT/salamandra-2b-instruct/blob/main/config.json
    dict(
        name="salamandra-2b{}",
        hf_config=dict(org="BSC-LT", name="salamandra-2b{}"),
        block_size=8192,
        vocab_size=256000,
        padded_vocab_size=256000,
        n_layer=24,
        n_head=16,
        n_embd=2048,
        n_query_groups=16,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=5440,
        norm_eps=1e-5,
        rope_base=10000
    ),
    # https://huggingface.co/BSC-LT/salamandra-7b-instruct/blob/main/config.json
    dict(
        name="salamandra-7b{}",
        hf_config=dict(org="BSC-LT", name="salamandra-7b{}"),
        block_size=8192,
        vocab_size=256000,
        padded_vocab_size=256000,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        norm_eps=1e-6,
        rope_base=10000
    ),
]

for c in salamandra:
    for kind in ("", "-instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


###############
# SmolLM2
###############
smollm2 = [
    # https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json
    dict(
        name="SmolLM2-135M{}",
        hf_config=dict(org="HuggingFaceTB", name="SmolLM2-135M{}"),
        block_size=8192,
        vocab_size=49152,
        padded_vocab_size=49152,
        n_layer=30,
        n_head=9,
        n_embd=576,
        n_query_groups=3,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=1536,
        rope_base=100000,
        norm_eps=1e-5,
    ),
    # https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json
    dict(
        name="SmolLM2-360M{}",
        hf_config=dict(org="HuggingFaceTB", name="SmolLM2-360M{}"),
        block_size=8192,
        vocab_size=49152,
        padded_vocab_size=49152,
        n_layer=32,
        n_head=15,
        n_embd=960,
        n_query_groups=5,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=2560,
        rope_base=100000,
        norm_eps=1e-5,
    ),
    # https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/config.json
    dict(
        name="SmolLM2-1.7B{}",
        hf_config=dict(org="HuggingFaceTB", name="SmolLM2-1.7B{}"),
        block_size=8192,
        vocab_size=49152,
        padded_vocab_size=49152,
        n_layer=24,
        n_head=32,
        n_embd=2048,
        n_query_groups=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=8192,
        rope_base=130000,
        norm_eps=1e-5,
    ),
]

for c in smollm2:
    for kind in ("", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

###############
# DeepSeek R1 Distill
###############

r1_distill_llama = [
    # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/blob/main/config.json
    dict(
        name="R1-Distill-Llama-8B",
        hf_config=dict(org="deepseek-ai", name="DeepSeek-R1-Distill-Llama-8B"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
    # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/config.json
    dict(
        name="R1-Distill-Llama-70B",
        hf_config=dict(org="deepseek-ai", name="DeepSeek-R1-Distill-Llama-70B"),
        block_size=131072,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
        rope_adjustments=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_seq_len=8192)
    ),
]

configs.extend(r1_distill_llama)

name_to_config = {config["name"]: config for config in configs}
