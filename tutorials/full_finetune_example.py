import os
import lightning as L
from litgpt.data import Alpaca
from litgpt.tokenizer import Tokenizer
from litgpt.model import GPT, Config
from litgpt.utils import num_parameters, CycleIterator
import torch
import torch.nn as nn

"""
This script is meant to be the simplest possible starting point for full finetuning a GPT model using lightning fabric with code (not CLI).

- no checkpoints
- no out dir
- no precision
- no resume
- no train/eval args (or any args in general)
- no logger (only to terminal)
- no grad accumulation
and no other fancy stuff. 

To add all the above stuff, you can slowly add them in yourself by looking at the code in litgpt/finetune/full.py or the docs for litgpt/fabric.
"""

# training params/args
SEED = 1337
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" # try also "stabilityai/stablelm-base-alpha-3b"!
BATCH_SIZE = 4
LR_WARMUP_STEPS = 100
MAX_STEPS = 601

def validate(model, val_dataloader): 
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, targets = batch["input_ids"], batch["labels"]
            logits = model(input_ids)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss += nn.functional.cross_entropy(logits[..., :-1, :], targets[..., 1:])
    fabric.print(f"Validation loss: {loss/len(val_dataloader)}")

def train(fabric, model, optimizer, scheduler, train_dataloader, val_dataloader):

    for iter_num, batch in enumerate(train_dataloader):
        input_ids, targets = batch["input_ids"], batch["labels"]

        # get model preds (logits)
        logits = model(input_ids)
        logits = logits.reshape(-1, logits.size(-1))

        # get loss
        targets = targets.reshape(-1)
        loss = nn.functional.cross_entropy(logits[..., :-1, :], targets[..., 1:])

        # update weights
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # print train loss every 100 steps
        if iter_num % 100 == 0 or iter_num == 0:
            fabric.print(f"Train iter {iter_num} -  loss {loss}")

        # validate every 300 steps
        if iter_num % 300 == 0 or iter_num == 0:
            validate(model, val_dataloader)
            model.train()
        iter_num += 1

        if iter_num >= MAX_STEPS:
            break

def main(fabric):
    fabric.seed_everything(SEED)
    
    # setup data, make tokenizer and make dataloaders
    data = Alpaca()
    tokenizer = Tokenizer(checkpoint_dir=f"checkpoints/{MODEL_NAME}")
    data.connect(tokenizer=tokenizer, batch_size=BATCH_SIZE, max_seq_length=1024)
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # print how many steps in an epoch
    fabric.print(f"Steps in an epoch: {len(train_dataloader)}")

    # setup model
    config = Config.from_file(f"checkpoints/{MODEL_NAME}/model_config.yaml")
    model = GPT(config)
    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    model = fabric.setup(model)

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.02, betas=(0.9, 0.95))
    optimizer = fabric.setup_optimizers(optimizer)

    # setup lr scheduler
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / LR_WARMUP_STEPS)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(MAX_STEPS - LR_WARMUP_STEPS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[LR_WARMUP_STEPS])

    # Start training!!!
    train(fabric, model, optimizer, scheduler, train_dataloader, val_dataloader)
    
if __name__ == "__main__":

    # check that the model exists (downloaded to ./checkpoints/)
    if not os.path.exists(f"checkpoints/{MODEL_NAME}"):
        print(f"Model {MODEL_NAME} not found. Please download it using `litgpt download --repo {MODEL_NAME}`")
        exit()

    ### Setup and launch
    fabric = L.Fabric(devices="auto", strategy="auto")
    fabric.launch(main)


