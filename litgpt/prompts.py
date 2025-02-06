# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
import re
from abc import abstractmethod
from json import dumps
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union

import yaml

from litgpt.config import Config

if TYPE_CHECKING:
    from litgpt import Tokenizer


class PromptStyle:
    """Base interface for prompt styles."""

    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return ([tokenizer.eos_id],)

    @classmethod
    def from_name(cls, name: str) -> "PromptStyle":
        return prompt_styles[name]()

    @classmethod
    def from_config(cls, config: Config) -> "PromptStyle":
        return model_name_to_prompt_style(config.name)


class Default(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return ([tokenizer.eos_id],)


class Alpaca(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        if kwargs.get("input"):
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Input:\n{kwargs['input']}\n\n### Response:\n"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )


class FLAN(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )


class Longform(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )


class StableLMAlpha(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language"
            " model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do"
            " anything that could be considered harmful to the user.\n- StableLM is more than just an information"
            " source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to"
            f" participate in anything that could harm a human.<|USER|>{prompt}<|ASSISTANT|>"
        )

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|SYSTEM|>")],
            [tokenizer.token_to_id("<|ASSISTANT|>")],
            [tokenizer.token_to_id("<|USER|>")],
        )


class StableLMZephyr(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n"


class Falcon(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"{prompt}\nAnswer:"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
            [193, tokenizer.token_to_id("User")],  # 193: '\n'
        )


class Falcon3(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n"
    
    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|endoftext|>")],
        )


class Llama2FunctionCalling(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # Has to be before the llama config
        b_func, e_func = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # This is an example for how to format functions for the model
        function_metadata = {
            "function": "search_bing",
            "description": (
                "Search the web for content on Bing. This allows users to search online/the internet/the web for"
                " content."
            ),
            "arguments": [{"name": "query", "type": "string", "description": "The search query string"}],
        }

        system_prompt = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            "possible. Your only response should be JSON formatted functions"
        )
        # replace the curly braces with double curly braces to escape them
        function_list = dumps(function_metadata).replace("{", "{{").replace("}", "}}")
        return (
            f"{b_func}{function_list.strip()}{e_func}{b_inst}{b_sys}"
            f"{system_prompt.strip()}"
            f"{e_sys}{prompt}{e_inst}\n\n"
        )


class Llama2(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        return (
            f"{b_inst} {b_sys}You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
            " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
            " instead of answering something not correct. If you don't know the answer to a question, please don't"
            f" share false information.{e_sys} {prompt} {e_inst} "
        )


class Llama3(PromptStyle):
    def apply(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: str) -> str:

        default_system_prompt = "You are a helpful assistant."

        # https://github.com/meta-llama/llama3/blob/359887376f0aaf30e433f23e25df858d8c2a9833/llama/tokenizer.py#L202-L229
        if isinstance(prompt, str):
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{default_system_prompt}<|eot_id|>" # No newline
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>" # No newline
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif isinstance(prompt, list):

            def encode_header(role: str) -> List[str]:
                return [f"<|start_header_id|>{role}<|end_header_id|>\n\n"]

            def encode_message(message: Dict[str, str]) -> List[str]:
                tokens = encode_header(message["role"])
                # NOTE: Meta stripped this. I'm not sure I agree, but who am I to argue?
                tokens.append(message["content"].strip())
                tokens.append("<|eot_id|>")
                return tokens

            def has_system_prompt(messages: List[Dict[str, str]]) -> bool:
                return messages[0].get("role", "") == "system" if len(messages) else False

            tokens = ["<|begin_of_text|>"]
            if not has_system_prompt(prompt):
                tokens.extend(encode_message({"role": "system", "content": default_system_prompt}))
            for i, message in enumerate(prompt):
                if i != 0 and message["role"] == "system":
                    raise ValueError("'system' role is only allowed at the beginning of the conversation list.")
                if not message["role"] in ["assistant", "user", "system"]:
                    raise ValueError(f"Unknown role: '{message['role']}'. Supported roles are 'assistant', 'user', and 'system'.")
                tokens.extend(encode_message(message))
            tokens.extend(encode_header("assistant"))
            return "".join(tokens)
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|eot_id|>")],
        )

class R1Base(PromptStyle):
    def apply(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: str) -> str:
        default_system_prompt = ""

        bos_token = "<｜begin▁of▁sentence｜>"
        eos_token = ""

        if isinstance(prompt, str):
            return (
                f"{default_system_prompt}"
                f"<｜User｜>{prompt}"
                f"<｜Assistant｜>"  # Prepares for assistant response
            )
        elif isinstance(prompt, list):

            def encode_message(message: Dict[str, str]) -> str:
                role = message["role"]
                content = message["content"].strip()
                
                if role == "system":
                    return content  # System prompt is prepended at the start
                elif role == "user":
                    return f"<｜User｜>{content}"
                elif role == "assistant":
                    return f"<｜Assistant｜>{content}{eos_token}"
                else:
                    raise ValueError(f"Unknown role: '{role}'. Supported roles are 'assistant', 'user', and 'system'.")

            # Extract system prompt (if any)
            system_prompt = ""
            if prompt[0].get("role") == "system":
                system_prompt = prompt[0]["content"]
                prompt = prompt[1:]  # Remove system message from the list

            # Construct the formatted prompt
            formatted_prompt = system_prompt
            for message in prompt:
                formatted_prompt += encode_message(message)

            formatted_prompt += "<｜Assistant｜>"  # Prepares for assistant response
            return formatted_prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<｜end▁of▁sentence｜>")],
        )

class FreeWilly2(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "### System:\nThis is a system prompt, please behave and help the user.\n\n"
            "### User:\n"
            f"{prompt}\n\n"
            "### Assistant:\n"
        )


class Platypus(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"### Instruction:\n\n{prompt}\n\n### Response:\n"

class StableCode(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"###Instruction\n{prompt}###Response\n"


class CodeLlama(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # for CodeLLama, we don't set a default system prompt, but it is supported:
        # https://huggingface.co/blog/codellama#conversational-instructions
        # Mistral does not: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
        b_inst, e_inst = "[INST]", "[/INST]"
        return f"{b_inst} {prompt} {e_inst}"


class Phi1(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"{prompt}\n\nAnswer:"

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            [198, tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            # [198, 198],  # '\n', '\n'
        )


class Phi2(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"Instruct: {prompt}\nOutput:"


class Phi3(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f'<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n'

class Phi4(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f'<|im_start|>user<|im_sep|>{prompt}<|im_end|><|im_start|>assistant<|im_sep|>'

class TinyLlama(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "<|system|>\n"
            "You are a friendly chatbot who always gives helpful, detailed, and polite answers.</s>\n"
            "<|user|>\n"
            f"{prompt}</s>\n"
            "<|assistant|>\n"
        )


class Gemma(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


class OLMo(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"


class OpenCoder(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        system_message = "You are OpenCoder, created by OpenCoder Team."
        return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


class ChatML(PromptStyle):
    def __init__(self, system_message: str):
        self.system_message = system_message

    def apply(self, prompt: str, **kwargs: str) -> str:
        return f"<|im_start|>system\n{self.system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

class Qwen2_5(ChatML):
    def __init__(self):
        super().__init__("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

class Qwen2_5_Math(ChatML):
    def __init__(self):
        super().__init__("Please reason step by step, and put your final answer within \\boxed{}.")

class QwQ(ChatML):
    def __init__(self):
        super().__init__("You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.")

class SmolLM2(ChatML):
    def __init__(self):
        super().__init__("You are a helpful AI assistant named SmolLM, trained by Hugging Face")

class Salamandra(ChatML):
    def __init__(self):
        super().__init__("I am Salamandra, an AI language model developed at the Barcelona Supercomputing Centre (BSC) by the Language Technologies Unit. My knowledge base was last updated on August 2023. Today Date: 2024-09-30\nSoy Salamandra, un modelo lingüístico de IA desarrollado en el Barcelona Supercomputing Centre (BSC) por la Language Technologies Unit. Mi base de conocimientos se actualizó por última vez en agosto de 2023.\nSoc Salamandra, un model de llenguatge d'IA desenvolupat al Barcelona Supercomputing Centre (BSC) per la Language Technologies Unit.")


# Maps prompt style names to PromptStyle classes
prompt_styles: Dict[str, Type[PromptStyle]] = {
    # Dataset-specific prompt styles
    "alpaca": Alpaca,
    "flan": FLAN,
    "longform": Longform,
    # Model-specific prompt styles
    "stablelm-alpha": StableLMAlpha,
    "stablelm-zephyr": StableLMZephyr,
    "falcon": Falcon,
    "llama2-function-calling": Llama2FunctionCalling,
    "llama2": Llama2,
    "freewilly2": FreeWilly2,
    "platypus": Platypus,
    "stablecode": StableCode,
    "codellama": CodeLlama,
    "phi-1": Phi1,
    "phi-2": Phi2,
    "phi-3": Phi3,
    "phi-4": Phi4,
    "tinyllama": TinyLlama,
    "gemma": Gemma,
    "llama3": Llama3,
    "olmo": OLMo,
    "opencoder": OpenCoder,
    "qwen2.5": Qwen2_5,
    "qwen2.5-math": Qwen2_5_Math,
    "qwq": QwQ,
    "smollm2": SmolLM2,
    "salamandra": Salamandra,
}


def model_name_to_prompt_style(model_name: str) -> PromptStyle:
    if re.search(r"stablelm-tuned-alpha", model_name):
        return StableLMAlpha()
    if re.search(r"stablelm-zephyr-3b", model_name):
        return StableLMZephyr()
    if re.search("stablecode-instruct", model_name):
        return StableCode()
    if re.search(r"Falcon3.*-Instruct", model_name):
        return Falcon3()
    if re.search(r"falcon.*-instruct", model_name):
        return Falcon()
    if re.search("Llama-2-7b-chat-hf-function-calling-v2", model_name):
        return Llama2FunctionCalling()
    if re.search("Llama-2.*-chat", model_name):
        return Llama2()
    if re.search("Llama-3.*-Instruct", model_name):
        return Llama3()
    if re.search("Llama-3.*-Instruct-*", model_name):
        return Llama3()
    if re.search("R1", model_name):
        return R1Base()
    if re.search("FreeWilly2", model_name):
        return FreeWilly2()
    if re.search("Platypus", model_name):
        return Platypus()
    if re.search("CodeLlama|Mi[sx]tral.*Instruct", model_name):
        return CodeLlama()
    if re.search("phi-1", model_name):
        return Phi1()
    if re.search("phi-2", model_name):
        return Phi2()
    if re.search("Phi-3", model_name):
        return Phi3()
    if re.search("phi-4", model_name):
        return Phi4()
    if re.search(r"tiny-llama.*chat", model_name):
        return TinyLlama()
    if re.search(r"(Code)?Gemma.*-it", model_name):
        return Gemma()
    if re.search(r"OLMo.*-hf", model_name):
        return OLMo()
    if re.search(r"OpenCoder.*-Instruct", model_name):
        return OpenCoder()
    if re.search(r"Qwen2\.5-Math-.*", model_name):
        return Qwen2_5_Math()
    if re.search(r"Qwen2\.5-.*", model_name):
        return Qwen2_5()
    if re.search(r"QwQ-.*", model_name):
        return QwQ()
    if re.search(r"SmolLM2.*-Instruct", model_name):
        return SmolLM2()
    if re.search(r"salamandra-.*-instruct", model_name):
        return Salamandra()
    return Default()


def save_prompt_style(style: Union[str, PromptStyle], checkpoint_dir: Path) -> None:
    style = PromptStyle.from_name(style) if isinstance(style, str) else style
    cls = type(style)
    # Allow saving the full module path for user-defined prompt classes
    config = {"class_path": f"{cls.__module__}.{cls.__name__}"}
    with open(checkpoint_dir / "prompt_style.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config, file)


def load_prompt_style(checkpoint_dir: Path) -> PromptStyle:
    with open(checkpoint_dir / "prompt_style.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # Support loading the full module path for user-defined prompt classes
    full_module_path, cls_name = config["class_path"].rsplit(".", 1)
    module = importlib.import_module(full_module_path)
    cls = getattr(module, cls_name)
    return cls()


def has_prompt_style(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "prompt_style.yaml").is_file()
