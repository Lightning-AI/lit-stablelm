# Download Model Weights with LitGPT

LitGPT supports a variety of LLM architectures with publicly available weights. You can download model weights and access a list of supported models using the `litgpt download list` command.

&nbsp;


| Model | Model size | Author | Reference |
|----|----|----|----|
| CodeGemma | 7B | Google | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma)                                                                 |
| Code Llama | 7B, 13B, 34B, 70B | Meta AI | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950)                                                                   |
| Danube2 | 1.8B | H2O.ai | [H2O.ai](https://h2o.ai/platform/danube-1-8b/)                                                                                             |
| Dolly | 3B, 7B, 12B | Databricks | [Conover et al. 2023](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)      |
| Falcon | 7B, 40B, 180B | TII UAE | [TII 2023](https://falconllm.tii.ae)                                                                                              |
| Falcon 3 | 1B, 3B, 7B, 10B | TII UAE | [TII 2024](https://huggingface.co/blog/falcon3)                                                                                              |
| FreeWilly2 (Stable Beluga 2) | 70B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models)                 |
| Function Calling Llama 2 | 7B | Trelis | [Trelis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)                                  |
| Gemma | 2B, 7B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                                       |
| Gemma 2 | 2B, 9B, 27B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)                              |
| Llama 2 | 7B, 13B, 70B | Meta AI | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)                                                                           |
| Llama 3 | 8B, 70B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3)                                                                                   |
| Llama 3.1 | 8B, 70B, 405B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3)                                                                           |
| Llama 3.2 | 1B, 3B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)                                    |
| Llama 3.3 | 70B | Meta AI | [Meta AI 2024](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)                                                                                 |
| Llama 3.1 Nemotron | 70B | NVIDIA | [NVIDIA AI 2024](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct/modelcard) |
| LongChat | 7B, 13B | LMSYS | [LongChat Team 2023](https://lmsys.org/blog/2023-06-29-longchat/)                                                                       |
| Mathstral | 7B | Mistral AI | [Mistral AI 2024](https://mistral.ai/news/mathstral/)                                                                        |
| MicroLlama | 300M | Ken Wang | [MicroLlama repo](https://github.com/keeeeenw/MicroLlama)
| Mixtral MoE | 8x7B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/mixtral-of-experts/)                                                                     |
| Mistral | 7B, 123B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/announcing-mistral-7b/)                                                                        |
| Mixtral MoE | 8x22B | Mistral AI | [Mistral AI 2024](https://mistral.ai/news/mixtral-8x22b/)                                                                         |
| Nous-Hermes | 7B, 13B, 70B | NousResearch | [Org page](https://huggingface.co/NousResearch)                                                                          |
| OLMo | 1B, 7B | Allen Institute for AI (AI2) | [Groeneveld et al. 2024](https://aclanthology.org/2024.acl-long.841/)     |
| OpenLLaMA | 3B, 7B, 13B | OpenLM Research | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)                                                         |
| Phi 1.5 & 2 | 1.3B, 2.7B | Microsoft Research  | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                          |
| Phi 3 & 3.5 | 3.8B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2404.14219)
| Phi 4 | 14B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2412.08905)                                                                            |
| Platypus | 7B, 13B, 70B |  Lee et al. | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317)                                                               |
| Pythia | {14,31,70,160,410}M, {1,1.4,2.8,6.9,12}B | EleutherAI | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)                                            |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | Alibaba Group | [Qwen Team 2024](https://qwenlm.github.io/blog/qwen2.5/)                                               |
| Qwen2.5 Coder | 0.5B, 1.5B, 3B, 7B, 14B, 32B | Alibaba Group | [Hui, Binyuan et al. 2024](https://arxiv.org/abs/2409.12186)                                          |
| Qwen2.5 1M (Long Context) | 7B, 14B | Alibaba Group | [Qwen Team 2025](https://qwenlm.github.io/blog/qwen2.5-1m/)                                          |
| Qwen2.5 Math | 1.5B, 7B, 72B | Alibaba Group | [An, Yang et al. 2024](https://arxiv.org/abs/2409.12122)                                          |
| QwQ | 32B | Alibaba Group | [Qwen Team 2024](https://qwenlm.github.io/blog/qwq-32b-preview/)                                                                         |
| R1 Distll Llama | 8B, 70B | DeepSeek AI | [DeepSeek AI 2025](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)                                                                         |
| RedPajama-INCITE | 3B, 7B | Together | [Together 2023](https://together.ai/blog/redpajama-models-v1)                                                                 |
| SmolLM2 | 135M, 360M, 1.7B | Hugging Face | [Hugging Face 2024](https://github.com/huggingface/smollm)                                                               |
| StableCode | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                                  |
| Salamandra | 2B, 7B | Barcelona Supercomputing Centre | [BSC-LTC 2024](https://github.com/BSC-LTC/salamandra)                                                                         |
| StableLM  | 3B, 7B | Stability AI | [Stability AI 2023](https://github.com/Stability-AI/StableLM)                                                                    |
| StableLM Zephyr | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                             |
| TinyLlama | 1.1B | Zhang et al. | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama)                                                                         |
| Vicuna | 7B, 13B, 33B | LMSYS | [Li et al. 2023](https://lmsys.org/blog/2023-03-30-vicuna/)                                                                          |                                                            |

&nbsp;

## General Instructions

### 1. List Available Models

To see all supported models, run the following command:

```bash
litgpt download list
```

The output is shown below:

```
allenai/OLMo-1B-hf
allenai/OLMo-7B-hf
allenai/OLMo-7B-Instruct-hf
bsc-lt/salamandra-2b
bsc-lt/salamandra-2b-instruct
bsc-lt/salamandra-7b
bsc-lt/salamandra-7b-instruct
codellama/CodeLlama-13b-hf
codellama/CodeLlama-13b-Instruct-hf
codellama/CodeLlama-13b-Python-hf
codellama/CodeLlama-34b-hf
codellama/CodeLlama-34b-Instruct-hf
codellama/CodeLlama-34b-Python-hf
codellama/CodeLlama-70b-hf
codellama/CodeLlama-70b-Instruct-hf
codellama/CodeLlama-70b-Python-hf
codellama/CodeLlama-7b-hf
codellama/CodeLlama-7b-Instruct-hf
codellama/CodeLlama-7b-Python-hf
databricks/dolly-v2-12b
databricks/dolly-v2-3b
databricks/dolly-v2-7b
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
deepseek-ai/DeepSeek-R1-Distill-Llama-70B
EleutherAI/pythia-1.4b
EleutherAI/pythia-1.4b-deduped
EleutherAI/pythia-12b
EleutherAI/pythia-12b-deduped
EleutherAI/pythia-14m
EleutherAI/pythia-160m
EleutherAI/pythia-160m-deduped
EleutherAI/pythia-1b
EleutherAI/pythia-1b-deduped
EleutherAI/pythia-2.8b
EleutherAI/pythia-2.8b-deduped
EleutherAI/pythia-31m
EleutherAI/pythia-410m
EleutherAI/pythia-410m-deduped
EleutherAI/pythia-6.9b
EleutherAI/pythia-6.9b-deduped
EleutherAI/pythia-70m
EleutherAI/pythia-70m-deduped
garage-bAInd/Camel-Platypus2-13B
garage-bAInd/Camel-Platypus2-70B
garage-bAInd/Platypus-30B
garage-bAInd/Platypus2-13B
garage-bAInd/Platypus2-70B
garage-bAInd/Platypus2-70B-instruct
garage-bAInd/Platypus2-7B
garage-bAInd/Stable-Platypus2-13B
google/codegemma-7b-it
google/gemma-2-27b
google/gemma-2-27b-it
google/gemma-2-2b
google/gemma-2-2b-it
google/gemma-2-9b
google/gemma-2-9b-it
google/gemma-2b
google/gemma-2b-it
google/gemma-7b
google/gemma-7b-it
h2oai/h2o-danube2-1.8b-chat
HuggingFaceTB/SmolLM2-135M
HuggingFaceTB/SmolLM2-135M-Instruct
HuggingFaceTB/SmolLM2-360M
HuggingFaceTB/SmolLM2-360M-Instruct
HuggingFaceTB/SmolLM2-1.7B
HuggingFaceTB/SmolLM2-1.7B-Instruct
lmsys/longchat-13b-16k
lmsys/longchat-7b-16k
lmsys/vicuna-13b-v1.3
lmsys/vicuna-13b-v1.5
lmsys/vicuna-13b-v1.5-16k
lmsys/vicuna-33b-v1.3
lmsys/vicuna-7b-v1.3
lmsys/vicuna-7b-v1.5
lmsys/vicuna-7b-v1.5-16k
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-13b-hf
meta-llama/Llama-2-70b-chat-hf
meta-llama/Llama-2-70b-hf
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-3.2-1B
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Llama-3.3-70B-Instruct
meta-llama/Meta-Llama-3-70B
meta-llama/Meta-Llama-3-70B-Instruct
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3-8B-Instruct
meta-llama/Meta-Llama-3.1-405B
meta-llama/Meta-Llama-3.1-405B-Instruct
meta-llama/Meta-Llama-3.1-70B
meta-llama/Meta-Llama-3.1-70B-Instruct
meta-llama/Meta-Llama-3.1-8B
meta-llama/Meta-Llama-3.1-8B-Instruct
microsoft/phi-1_5
microsoft/phi-2
microsoft/Phi-3-mini-128k-instruct
microsoft/Phi-3-mini-4k-instruct
microsoft/Phi-3.5-mini-instruct
microsoft/phi-4
mistralai/mathstral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.1
mistralai/Mistral-7B-Instruct-v0.2
mistralai/Mistral-7B-Instruct-v0.3
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-v0.3
mistralai/Mistral-Large-Instruct-2407
mistralai/Mistral-Large-Instruct-2411
mistralai/Mixtral-8x7B-Instruct-v0.1
mistralai/Mixtral-8x7B-v0.1
mistralai/Mixtral-8x22B-Instruct-v0.1
mistralai/Mixtral-8x22B-v0.1
NousResearch/Nous-Hermes-13b
NousResearch/Nous-Hermes-llama-2-7b
NousResearch/Nous-Hermes-Llama2-13b
nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
openlm-research/open_llama_13b
openlm-research/open_llama_3b
openlm-research/open_llama_7b
Qwen/Qwen2.5-0.5B
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-7B-Instruct-1M
Qwen/Qwen2.5-14B
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-14B-Instruct-1M
Qwen/Qwen2.5-32B
Qwen/Qwen2.5-32B-Instruct
Qwen/Qwen2.5-72B
Qwen/Qwen2.5-72B-Instruct
Qwen/Qwen2.5-Coder-0.5B
Qwen/Qwen2.5-Coder-0.5B-Instruct
Qwen/Qwen2.5-Coder-1.5B
Qwen/Qwen2.5-Coder-1.5B-Instruct
Qwen/Qwen2.5-Coder-3B
Qwen/Qwen2.5-Coder-3B-Instruct
Qwen/Qwen2.5-Coder-7B
Qwen/Qwen2.5-Coder-7B-Instruct
Qwen/Qwen2.5-Coder-14B
Qwen/Qwen2.5-Coder-14B-Instruct
Qwen/Qwen2.5-Coder-32B
Qwen/Qwen2.5-Coder-32B-Instruct
Qwen/Qwen2.5-Math-1.5B
Qwen/Qwen2.5-Math-1.5B-Instruct
Qwen/Qwen2.5-Math-7B
Qwen/Qwen2.5-Math-7B-Instruct
Qwen/Qwen2.5-Math-72B
Qwen/Qwen2.5-Math-72B-Instruct
Qwen/QwQ-32B-Preview
stabilityai/FreeWilly2
stabilityai/stable-code-3b
stabilityai/stablecode-completion-alpha-3b
stabilityai/stablecode-completion-alpha-3b-4k
stabilityai/stablecode-instruct-alpha-3b
stabilityai/stablelm-3b-4e1t
stabilityai/stablelm-base-alpha-3b
stabilityai/stablelm-base-alpha-7b
stabilityai/stablelm-tuned-alpha-3b
stabilityai/stablelm-tuned-alpha-7b
stabilityai/stablelm-zephyr-3b
tiiuae/falcon-180B
tiiuae/falcon-180B-chat
tiiuae/falcon-40b
tiiuae/falcon-40b-instruct
tiiuae/falcon-7b
tiiuae/falcon-7b-instruct
tiiuae/Falcon3-1B-Base
tiiuae/Falcon3-1B-Instruct
tiiuae/Falcon3-3B-Base
tiiuae/Falcon3-3B-Instruct
tiiuae/Falcon3-7B-Base
tiiuae/Falcon3-7B-Instruct
tiiuae/Falcon3-10B-Base
tiiuae/Falcon3-10B-Instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
togethercomputer/LLaMA-2-7B-32K
togethercomputer/RedPajama-INCITE-7B-Base
togethercomputer/RedPajama-INCITE-7B-Chat
togethercomputer/RedPajama-INCITE-7B-Instruct
togethercomputer/RedPajama-INCITE-Base-3B-v1
togethercomputer/RedPajama-INCITE-Base-7B-v0.1
togethercomputer/RedPajama-INCITE-Chat-3B-v1
togethercomputer/RedPajama-INCITE-Chat-7B-v0.1
togethercomputer/RedPajama-INCITE-Instruct-3B-v1
togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1
Trelis/Llama-2-7b-chat-hf-function-calling-v2
unsloth/Mistral-7B-v0.2
```

&nbsp;

> [!TIP]
> To sort the list above by model name after the `/`, use `litgpt download list | sort -f -t'/' -k2`.

&nbsp;

> [!NOTE]
> If you want to adopt a model variant that is not listed in the table above but has a similar architecture as one of the supported models, you can use this model by by using the `--model_name` argument as shown below:
>
> ```bash
> litgpt download NousResearch/Hermes-2-Pro-Mistral-7B \
>  --model_name Mistral-7B-v0.1
> ```

&nbsp;

### 2. Download Model Weights

To download the weights for a specific model provide a `<repo_id>` with the model's repository ID. For example:

```bash
litgpt download <repo_id>
```

This command downloads the model checkpoint into the `checkpoints/` directory.

&nbsp;

### 3. Additional Help

For more options, add the `--help` flag when running the script:

```bash
litgpt download --help
```

&nbsp;

### 4. Run the Model

After conversion, run the model with the given checkpoint path as input, adjusting `repo_id` accordingly:

```bash
litgpt chat <repo_id>
```

&nbsp;

## Tinyllama Example

This section shows a typical end-to-end example for downloading and using TinyLlama:

1. List available TinyLlama checkpoints:

```bash
litgpt download list | grep Tiny
```

```
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

2. Download a TinyLlama checkpoint:

```bash
export repo_id=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
litgpt download $repo_id
```

3. Use the TinyLlama model:

```bash
litgpt chat $repo_id
```

&nbsp;
## Specific models and access tokens

Note that certain models require that you've been granted access to the weights on the Hugging Face Hub.

For example, to get access to the Gemma 2B model, you can do so by following the steps at <https://huggingface.co/google/gemma-2b>. After access is granted, you can find your HF hub token in <https://huggingface.co/settings/tokens>.

Once you've been granted access and obtained the access token you need to pass the additional `--access_token`:

```bash
litgpt download google/gemma-2b \
  --access_token your_hf_token
```

&nbsp;

## Finetunes and Other Model Variants

Sometimes you want to download the weights of a finetune of one of the models listed above. To do this, you need to manually specify the `model_name` associated to the config to use. For example:

```bash
litgpt download NousResearch/Hermes-2-Pro-Mistral-7B \
  --model_name Mistral-7B-v0.1
```

&nbsp;

## Tips for GPU Memory Limitations

The `litgpt download` command will automatically convert the downloaded model checkpoint into a LitGPT-compatible format. In case this conversion fails due to GPU memory constraints, you can try to reduce the memory requirements by passing the  `--dtype bf16-true` flag to convert all parameters into this smaller precision (however, note that most model weights are already in a bfloat16 format, so it may not have any effect):

```bash
litgpt download <repo_id>
  --dtype bf16-true
```

(If your GPU does not support the bfloat16 format, you can also try a regular 16-bit float format via `--dtype 16-true`.)

&nbsp;

## Converting Checkpoints Manually

For development purposes, for example, when adding or experimenting with new model configurations, it may be beneficial to split the weight download and model conversion into two separate steps.

You can do this by passing the `--convert_checkpoint false` option to the download script:

```bash
litgpt download <repo_id> \
  --convert_checkpoint false
```

and then calling the `convert_hf_checkpoint` command:

```bash
litgpt convert_to_litgpt <repo_id>
```

&nbsp;

## Downloading Tokenizers Only

In some cases we don't need the model weight, for example, when we are pretraining a model from scratch instead of finetuning it. For cases like this, you can use the `--tokenizer_only` flag to only download a model's tokenizer, which can then be used in the pretraining scripts:

```bash
litgpt download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --tokenizer_only true
```

and

```bash
litgpt pretrain tiny-llama-1.1b \
  --data ... \
  --tokenizer_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/
```
