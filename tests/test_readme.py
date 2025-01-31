# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
import requests
from tests.conftest import RunIf

REPO_ID = Path("EleutherAI/pythia-14m")
CUSTOM_TEXTS_DIR = Path("custom_texts")


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Command '{' '.join(command)}' failed with exit status {e.returncode}\n"
            f"Output:\n{e.stdout}\n"
            f"Error:\n{e.stderr}"
        )
        # You can either print the message, log it, or raise an exception with it
        print(error_message)
        raise RuntimeError(error_message) from None


@pytest.mark.dependency()
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_download_model():
    repo_id = str(REPO_ID).replace("\\", "/")  # fix for Windows CI
    command = ["litgpt", "download", str(repo_id)]
    output = run_command(command)

    s = Path("checkpoints") / repo_id
    assert f"Saving converted checkpoint to {str(s)}" in output
    assert ("checkpoints" / REPO_ID).exists()
    assert ("checkpoints" / REPO_ID / "pytorch_model.bin").exists(), f"{REPO_ID} not downloaded"

    # Also test valid but unsupported repo IDs
    command = ["litgpt", "download", "CohereForAI/aya-23-8B"]
    output = run_command(command)
    assert "Unsupported `repo_id`" in output


@pytest.mark.dependency()
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_download_books():
    CUSTOM_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    books = [
        ("https://www.gutenberg.org/cache/epub/24440/pg24440.txt", "book1.txt"),
        ("https://www.gutenberg.org/cache/epub/26393/pg26393.txt", "book2.txt")
    ]
    for url, filename in books:
        subprocess.run(["curl", url, "--output", str(CUSTOM_TEXTS_DIR / filename)], check=True)
        # Verify each book is downloaded
        assert (CUSTOM_TEXTS_DIR / filename).exists(), f"{filename} not downloaded"


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model"])
def test_chat_with_model():
    command = ["litgpt", "generate", "checkpoints" / REPO_ID]
    prompt = "What do Llamas eat?"
    result = subprocess.run(command, input=prompt, text=True, capture_output=True, check=True)
    assert "What food do llamas eat?" in result.stdout


@RunIf(min_cuda_gpus=1)
@pytest.mark.dependency(depends=["test_download_model"])
def test_chat_with_quantized_model():
    command = ["litgpt", "generate", "checkpoints" / REPO_ID, "--quantize", "bnb.nf4", "--precision", "bf16-true"]
    prompt = "What do Llamas eat?"
    result = subprocess.run(command, input=prompt, text=True, capture_output=True, check=True)
    assert "What food do llamas eat?" in result.stdout, result.stdout


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model"])
@pytest.mark.timeout(300)
def test_finetune_model(tmp_path):

    OUT_DIR = tmp_path / "out" / "lora"
    DATASET_PATH = tmp_path / "custom_finetuning_dataset.json"
    CHECKPOINT_DIR = "checkpoints" / REPO_ID

    download_command = ["curl", "-L", "https://huggingface.co/datasets/medalpaca/medical_meadow_health_advice/raw/main/medical_meadow_health_advice.json", "-o", str(DATASET_PATH)]
    subprocess.run(download_command, check=True)

    assert DATASET_PATH.exists(), "Dataset file not downloaded"

    finetune_command = [
        "litgpt", "finetune_lora",
        str(CHECKPOINT_DIR),
        "--lora_r", "1",
        "--data", "JSON",
        "--data.json_path", str(DATASET_PATH),
        "--data.val_split_fraction", "0.00001",  # Keep small because new final validation is expensive
        "--train.max_steps", "1",
        "--out_dir", str(OUT_DIR)
    ]
    run_command(finetune_command)

    generated_out_dir = OUT_DIR/"final"
    assert generated_out_dir.exists(), f"Finetuning output directory ({generated_out_dir}) was not created"
    model_file = OUT_DIR/"final"/"lit_model.pth"
    assert model_file.exists(), f"Model file ({model_file}) was not created"


@pytest.mark.skipif(
    sys.platform.startswith("win") or
    sys.platform == "darwin",
    reason="`torch.compile` is not supported on this OS."
)
@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model", "test_download_books"])
def test_pretrain_model(tmp_path):
    OUT_DIR = tmp_path / "out" / "custom_pretrained"
    pretrain_command = [
        "litgpt", "pretrain",
        "pythia-14m",
        "--tokenizer_dir", str("checkpoints" / REPO_ID),
        "--data", "TextFiles",
        "--data.train_data_path", str(CUSTOM_TEXTS_DIR),
        "--train.max_tokens", "100",     # to accelerate things for CI
        "--eval.max_iters", "1",         # to accelerate things for CI
        "--out_dir", str(OUT_DIR)
    ]
    output = run_command(pretrain_command)

    assert "Warning: Preprocessed training data found" not in output
    out_dir_path = OUT_DIR / "final"
    assert out_dir_path.exists(), f"Pretraining output directory ({out_dir_path}) was not created"
    out_model_path = OUT_DIR / "final" / "lit_model.pth"
    assert out_model_path.exists(), f"Model file ({out_model_path}) was not created"

    # Test that warning is displayed when running it a second time
    output = run_command(pretrain_command)
    assert "Warning: Preprocessed training data found" in output


@pytest.mark.skipif(
    sys.platform.startswith("win") or
    sys.platform == "darwin",
    reason="`torch.compile` is not supported on this OS."
)
@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
@pytest.mark.dependency(depends=["test_download_model", "test_download_books"])
def test_continue_pretrain_model(tmp_path):
    OUT_DIR = tmp_path / "out" / "custom_continue_pretrained"
    pretrain_command = [
        "litgpt", "pretrain",
        "pythia-14m",
        "--initial_checkpoint", str("checkpoints" / REPO_ID),
        "--tokenizer_dir", str("checkpoints" / REPO_ID),
        "--data", "TextFiles",
        "--data.train_data_path", str(CUSTOM_TEXTS_DIR),
        "--train.max_tokens", "100",     # to accelerate things for CI
        "--eval.max_iters", "1",         # to accelerate things for CI
        "--out_dir", str(OUT_DIR)
    ]
    run_command(pretrain_command)

    generated_out_dir = OUT_DIR/"final"
    assert generated_out_dir.exists(), f"Continued pretraining directory ({generated_out_dir}) was not created"
    model_file = OUT_DIR/"final"/"lit_model.pth"
    assert model_file.exists(), f"Model file ({model_file}) was not created"


@pytest.mark.dependency(depends=["test_download_model"])
def test_serve():
    CHECKPOINT_DIR = str("checkpoints" / REPO_ID)
    run_command = [
        "litgpt", "serve", "--devices", "0", str(CHECKPOINT_DIR)
    ]

    process = None

    def run_server():
        nonlocal process

        stdout, stderr = ""
        try:
            print (f"Running command {run_command}")
            process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=90)
            print (f"stdout : {stdout}")
            print(f"stderr : {stderr}")
        except subprocess.TimeoutExpired as te:
            print('Server start-up timeout expired')
            raise te
        except Exception as e:
            print (e)
            raise e
        finally:
            print (f"stdout : {stdout}")
            print(f"stderr : {stderr}")

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    # Allow time to initialize and start serving
    time.sleep(30)

    try:
        response = requests.get("http://127.0.0.1:8000")
        print(response.status_code)
        print (response.text)
        assert response.status_code == 200, "Server did not respond as expected."
    except Exception as e:
            print (e)
            raise e
    finally:
        if process:
            process.kill()
        server_thread.join()
