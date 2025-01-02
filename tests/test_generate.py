# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from unittest import mock
from unittest.mock import ANY, Mock, call

import pytest
import torch
import yaml

import litgpt.generate.base as generate
from litgpt import GPT, Config
from litgpt.generate.base import sample


skip_in_ci_on_macos = pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipped on macOS in CI environment because CI machine does not have enough memory to run this test."
)


def test_generate():
    import lightning as L
    L.seed_everything(1234)

    T = 5
    input_idx = torch.arange(0, T)

    config = Config(
        block_size=128,
        vocab_size=16,
        n_layer=1,
        n_head=4,
        n_embd=8,
    )
    model = GPT(config)
    max_new_tokens = 20
    model.max_seq_length = T + max_new_tokens
    model.set_kv_cache(batch_size=1)

    multinomial_results = []

    def multinomial(*args, **kwargs):
        if args:
            probs = args[0]
        else:
            probs = kwargs.get("probs")
        res_shape, fin_dim = probs.shape[:-1], probs.shape[-1]
        if probs.ndim > 2:
            probs = probs.view(-1, fin_dim)
        out = torch.multinomial(probs, num_samples=1).view(*res_shape)
        multinomial_results.append(out)
        return out

    with mock.patch("litgpt.generate.base.multinomial_num_samples_1", multinomial):
        out = generate.generate(
            model=model,
            prompts=[input_idx],
            max_returned_tokens=T + max_new_tokens,
            top_k=1,
        )[0]

    assert out.size(0) == T + max_new_tokens, (out.size(0), T + max_new_tokens)
    multinomial_results = torch.hstack(multinomial_results)
    print(f"input_idx {input_idx.shape}, multinomial_results: {multinomial_results.shape}")
    expected = torch.cat((input_idx, multinomial_results.squeeze(0)))
    assert out.shape == expected.shape, (out.shape, expected.shape)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "max_seq_length",
    (
        10,
        20 + 5,
        128,
    ),
)
def test_generate_single_vs_batch(max_seq_length):
    import lightning as L
    L.seed_everything(1234)

    max_prompt_size = int(max_seq_length * 0.75)
    num_prompts = 6
    vocab_size = 128
    prompts = [
        torch.randint(
            low=0,
            high=vocab_size,
            size=(torch.randint(low=1, high=max_prompt_size, size = (1,)).item(),)
        )
        for _ in range(num_prompts)
    ]
    print(f"max_seq_length = {max_seq_length}")

    config = Config(
        block_size=128,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=8,
        rotary_percentage=1,
    )
    model = GPT(config)
    model.max_seq_length = max_seq_length

    res_batch = generate.generate(
        model=model,
        prompts=prompts,
        max_returned_tokens=max_seq_length,
        top_k=1,
    )
    res_single = [
        generate.generate(
            model=model,
            prompts=[prompt],
            max_returned_tokens=max_seq_length,
            top_k=1,
        )[0]
        for prompt in prompts
    ]

    for rb, rs, prompt in zip(res_batch, res_single, prompts):
        print(f"rs: {rs}\nrb: {rb}\npr: {prompt}")
        torch.testing.assert_close(rs, rb)
        print("OK")


@skip_in_ci_on_macos
def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    config_path = fake_checkpoint_dir / "model_config.yaml"
    config = {
        "block_size": 128,
        "vocab_size": 50,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 8,
        "rotary_percentage": 1,
    }
    config_path.write_text(yaml.dump(config))

    module_mock = Mock()
    module_mock.config.block_size = config["block_size"]
    load_mock = Mock()
    load_mock.return_value = load_mock
    monkeypatch.setattr(generate, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    tokenizer_mock.return_value.eos_id.return_value = 255  # TODO (does not work)
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor(
        [
            1, 2, 3, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 1, 1, 0
        ]
    )
    monkeypatch.setattr(generate, "generate", generate_mock)

    num_samples = 2
    out, err = StringIO(), StringIO()
    sample_kwargs = dict(
        temperature=2.0,
        top_k=2,
        top_p=0.9,
    )
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(
            **sample_kwargs,
            num_samples=num_samples,
            checkpoint_dir=fake_checkpoint_dir,
        )

    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(
        tokenizer_mock.return_value.decode.call_args[0][0].to(torch.device("cpu")),
        generate_mock.return_value
    )
    assert (
        generate_mock.mock_calls
        == [call(ANY, tensor_like, 53, **sample_kwargs, eos_id=tokenizer_mock.return_value.eos_id)]
        * num_samples
    )
    expected_output = "foo bar baz\n" * num_samples
    # Allow for the config to be printed before the expected repeated strings.
    pattern = rf".*^{re.escape(expected_output.strip())}$.*"
    assert re.match(pattern, out.getvalue().strip(), re.DOTALL | re.MULTILINE)

    err_value = err.getvalue()
    expected_parts = [
        "'padded_vocab_size': 512",
        "'n_layer': 2",
        "'n_head': 4",
    ]
    assert all(part in err_value for part in expected_parts)


def test_cli():
    args = ["litgpt", "generate", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Default generation option" in output


@pytest.mark.parametrize("temperature", (0.0, 1.0, 0.5))
def test_sample(temperature):
    # shape: 2x3x5
    logits = torch.tensor(
        [
            [[24, 4, 98, 77, 47], [65, 70, 32, 67, 24], [92, 32, 88, 36, 62]],
            [[85, 79, 57, 68, 50], [89, 46, 72, 45, 32], [68, 96, 68, 24, 36]],
        ],
        dtype=torch.float32,
    )
    token = sample(logits, temperature=temperature, top_p=0.8)

    assert token.shape == (2, 3)
    # sample is batch size 1 only for now - this should be [0, 1] once batched generation is supported
    assert token[0, -1].item() == 0


def test_generate_different_results_with_different_top_p():
    config = Config(
        block_size=128,
        vocab_size=16,
        n_layer=1,
        n_head=4,
        n_embd=8,
        rotary_percentage=1,
    )
    model = GPT(config)
    model.max_seq_length = 50
    model.set_kv_cache(batch_size=1)

    torch.manual_seed(123)
    input_idx = torch.randint(10, size=(1,))

    torch.manual_seed(123)
    output1 = generate.generate(
        model=model,
        prompts=[input_idx],
        max_returned_tokens=20,
        top_p=1.0,
    )[0]
    torch.manual_seed(123)
    output2 = generate.generate(
        model=model,
        prompts=[input_idx],
        max_returned_tokens=20,
        top_p=0.1,
    )[0]

    assert not torch.equal(output1, output2)
