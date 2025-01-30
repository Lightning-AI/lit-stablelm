# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import math
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union
import warnings

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import RunningMean

from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca, DataModule
from litgpt.generate.base import generate
from litgpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from litgpt.prompts import save_prompt_style
from litgpt.scripts.merge_lora import merge_lora
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_nvlink_connectivity,
    create_finetuning_performance_report,
    CycleIterator,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    load_checkpoint,
    init_out_dir,
    instantiate_torch_optimizer,
    instantiate_bnb_optimizer,
    num_parameters,
    parse_devices,
    save_hyperparameters,
    select_sft_generate_example,
)


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/lora"),
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = False,
    lora_value: bool = True,
    lora_projection: bool = False,
    lora_mlp: bool = False,
    lora_head: bool = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=5,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(interval=100, max_new_tokens=100, max_iters=100),
    optimizer: Union[str, Dict] = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 1337,
    access_token: Optional[str] = None,
) -> None:
    """Finetune a model using the LoRA method.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        quantize: If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information.
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        lora_r: The LoRA rank.
        lora_alpha: The LoRA alpha.
        lora_dropout: The LoRA dropout value.
        lora_query: Whether to apply LoRA to the query weights in attention.
        lora_key: Whether to apply LoRA to the key weights in attention.
        lora_value: Whether to apply LoRA to the value weights in attention.
        lora_projection: Whether to apply LoRA to the output projection in the attention block.
        lora_mlp: Whether to apply LoRA to the weights of the MLP in the attention block.
        lora_head: Whether to apply LoRA to output head in GPT.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    pprint(locals())
    data = Alpaca() if data is None else data
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(
        checkpoint_dir / "model_config.yaml",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_query=lora_query,
        lora_key=lora_key,
        lora_value=lora_value,
        lora_projection=lora_projection,
        lora_mlp=lora_mlp,
        lora_head=lora_head,
    )

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(logger_name, out_dir, name=f"finetune-{config.name}", log_interval=train.log_interval)

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

    if devices * num_nodes > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 and num_nodes=1"
                " when using the --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={torch.nn.Linear},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
        plugins=plugins,
    )

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(main, devices, seed, config, data, checkpoint_dir, out_dir, train, eval, optimizer)


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        optimizer = instantiate_bnb_optimizer(optimizer, model.parameters())

        from bitsandbytes.nn import StableEmbedding
        old_embedding = model.transformer.wte
        model.transformer.wte = StableEmbedding(old_embedding.num_embeddings, old_embedding.embedding_dim)
        with torch.no_grad():
            model.transformer.wte.weight.copy_(old_embedding.weight)
        model.transformer.wte = model.transformer.wte.to(device=old_embedding.weight.device, dtype=old_embedding.weight.dtype)
    else:
        optimizer = instantiate_torch_optimizer(optimizer, model.parameters())

    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps)

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    train_time = time.perf_counter()
    token_counts = fit(
        fabric,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        devices,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        data,
    )

    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(training_time, token_counts, fabric.device.type)
    fabric.print(output)

    # Final evaluation
    if eval.final_validation:
        val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=len(val_dataloader)))
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics)
        fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth.lora"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_checkpoint(fabric, model, save_path)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data.prompt_style, save_path.parent)
        merge_lora(checkpoint_dir=save_path.parent)


def fit(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
) -> dict:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(ConcatDataset([train_dataloader.dataset, val_dataloader.dataset]))
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    if eval.initial_validation:
        val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=len(val_dataloader)))
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=2), verbose=False)  # sanity check
        val_loss = "n/a"

    train_iterator = CycleIterator(train_dataloader)
    throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    max_steps = train.max_steps or float("inf")
    step_count = 0
    iter_num = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    token_counts = {
        "raw_tokens": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template_and_padding": torch.tensor(0, device=fabric.device, dtype=torch.long),
    }

    while step_count < max_steps:
        iter_num += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        if train_iterator.epoch >= train.epochs:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_count += 1

        token_counts["raw_tokens"] += batch["token_counts"]["raw"].sum().item()
        token_counts["raw_tokens_plus_prompt_template"] += batch["token_counts"]["raw_plus_prompt_template"].sum().item()
        token_counts["raw_tokens_plus_prompt_template_and_padding"] += input_ids.numel()

        total_lengths += input_ids.numel()
        if iter_num % train.log_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0, batches=iter_num, samples=iter_num * train.micro_batch_size, lengths=total_lengths
            )
            throughput.compute_and_log(step=iter_num)
            metrics = {
                "loss": loss,
                "iter": iter_num,
                "step": step_count,
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": token_counts["raw_tokens_plus_prompt_template"],
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"] * fabric.world_size,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=iter_num)

        if not is_accumulating and step_count % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, eval)
            generate_example(fabric, model, tokenizer, eval, data)
            t1 = time.perf_counter() - t0
            fabric.print(f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=iter_num)
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and step_count % train.save_interval == 0:
            checkpoint_file = out_dir / f"step-{step_count:06d}" / "lit_model.pth.lora"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            save_lora_checkpoint(fabric, model, checkpoint_file)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)

    total_token_counts = {}
    for key in token_counts:
        total = fabric.all_reduce(token_counts[key], reduce_op="sum")
        total_token_counts[key] = total.item()

    return total_token_counts


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: GPT, val_dataloader: DataLoader, eval: EvalArgs, verbose: bool = True) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)

    val_loss = losses.mean()

    model.train()
    return val_loss


@torch.no_grad()
def generate_example(fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, eval: EvalArgs, data: DataModule):
    instruction = select_sft_generate_example(eval, data)

    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    model.eval()

    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < model.max_seq_length:
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model, encoded, max_returned_tokens=max_returned_tokens, temperature=0.8, eos_id=tokenizer.eos_id
        )
        model.clear_kv_cache()
        model.train()
        output = tokenizer.decode(output)
        fabric.print(f"{output}\n")
    else:
        print(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {model.config.block_size}."
        )


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=train.max_seq_length)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if not train.epochs and not train.max_steps:
        issues.append(f"{__file__} requires either epochs or max_steps to be set. This is set in {train}")
    if issues:
        raise ValueError("\n".join(issues))
