# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from pathlib import Path
from pprint import pprint
from typing import Dict, Any, Optional, Literal

from lightning_utilities.core.imports import RequirementCache
import torch

from litgpt.api import LLM

from litgpt.utils import auto_download_checkpoint

_LITSERVE_AVAILABLE = RequirementCache("litserve")
if _LITSERVE_AVAILABLE:
    from litserve import LitAPI, LitServer, OpenAISpec, OpenAIEmbeddingSpec
    from litserve.specs.openai import ChatMessage
else:
    LitAPI, LitServer = object, object


class BaseLitAPI(LitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1,
    ) -> None:
        print ("BaseLitAPI : __init__")
        if not _LITSERVE_AVAILABLE:
            raise ImportError(str(_LITSERVE_AVAILABLE))

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.quantize = quantize
        self.precision = precision
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.devices = devices

    def setup(self, device: str) -> None:
        print ("BaseLitAPI : setup")
        if ":" in device:
            accelerator, device = device.split(":")
            device = f"[{int(device)}]"
        else:
            accelerator = device
            device = 1

        print("Initializing model...")
        self.llm = LLM.load(
            model=self.checkpoint_dir,
            distribute=None
        )

        self.llm.distribute(
            devices=self.devices,
            accelerator=accelerator,
            quantize=self.quantize,
            precision=self.precision,
            generate_strategy="sequential" if self.devices is not None and self.devices > 1 else None
        )
        print("Model successfully initialized.")

    def decode_request(self, request: Dict[str, Any]) -> Any:
        # Convert the request payload to your model input.
        print ("BaseLitAPI : decode_request")
        prompt = str(request["prompt"])
        return prompt


class SimpleLitAPI(BaseLitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[str] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1,
    ):
        super().__init__(checkpoint_dir, quantize, precision, temperature, top_k, top_p, max_new_tokens, devices)   

    def setup(self, device: str):
        super().setup(device)

    def predict(self, inputs: str) -> Any:
        output = self.llm.generate(
            inputs,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )
        return output

    def encode_response(self, output: str) -> Dict[str, Any]:
        # Convert the model output to a response payload.
        return {"output": output}


class StreamLitAPI(BaseLitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[str] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1
    ):
        print ("StreamLitAPI : __init__")
        super().__init__(checkpoint_dir, quantize, precision, temperature, top_k, top_p, max_new_tokens, devices)   

    def setup(self, device: str):
        print ("StreamLitAPI : setup")
        super().setup(device)

    def predict(self, inputs: torch.Tensor) -> Any:
        # Run the model on the input and return the output.
        print ("StreamLitAPI : predict")
        yield from self.llm.generate(
            inputs,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            stream=True
        )

    def encode_response(self, output):
        print ("StreamLitAPI : encode_response")
        for out in output:
            yield {"output": out}

class OpenAIAPI(LitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        quantize: Optional[str] = None,
        precision: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        devices: int = 1
    ):
        print ("OpenAIAPI : init")  
        if not _LITSERVE_AVAILABLE:
            raise ImportError(str(_LITSERVE_AVAILABLE))

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.quantize = quantize
        self.precision = precision
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.devices = devices

    def setup(self, device: str):
        print ("OpenAIAPI : setup")
        if ":" in device:
            accelerator, device = device.split(":")
            device = f"[{int(device)}]"
        else:
            accelerator = device
            device = 1

        print("OpenAIAPI : Initializing model...")
        self.llm = LLM.load(
            model=self.checkpoint_dir,
            distribute=None
        )

        self.llm.distribute(
            devices=self.devices,
            accelerator=accelerator,
            quantize=self.quantize,
            precision=self.precision,
            generate_strategy="sequential" if self.devices is not None and self.devices > 1 else None
        )
        print("OpenAIAPI : Model successfully initialized.")
    
    def predict(self, inputs: torch.Tensor) -> Any:
        # Run the model on the input and return the output.
        
        for chunk in inputs:
            yield from self.llm.generate(
            chunk["content"],
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            stream=True
            )


def run_server(
    checkpoint_dir: Path,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 50,
    devices: int = 1,
    accelerator: str = "auto",
    port: int = 8000,
    stream: bool = False,
    access_token: Optional[str] = None,
    spec: Optional[Literal["openaispec", "openaiembeddingspec"]] = None,
    
) -> None:
    """Serve a LitGPT model using LitServe.

    Evaluate a model with the LM Evaluation Harness.

    Arguments:
        checkpoint_dir: The checkpoint directory to load the model from.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
        temperature: Temperature setting for the text generation. Value above 1 increase randomness.
            Values below 1 decrease randomness.
        top_k: The size of the pool of potential next tokens. Values larger than 1 result in more novel
            generated text but can also lead to more incoherent texts.
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
        max_new_tokens: The number of generation steps to take.
        devices: How many devices/GPUs to use.
        accelerator: The type of accelerator to use. For example, "auto", "cuda", "cpu", or "mps".
            The "auto" setting (default) chooses a GPU if available, and otherwise uses a CPU.
        port: The network port number on which the model is configured to be served.
        stream: Whether to stream the responses.
        access_token: Optional API token to access models with restrictions.
        spec: Classname of the LitServe protocol spec to use. Options include: OpenAISpec, OpenAIEmbeddingSpec
        
            Specs defined in:
            https://github.com/Lightning-AI/LitServe/tree/main/src/litserve/specs
            OpenAI in LitServe details:
            https://lightning.ai/docs/litserve/features/open-ai-spec#enable-openai-api-in-litserve
    """
    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    pprint(locals())

    spec_impl = None
    if not stream:

        spec_impl = None

        if (spec != None):
            if (spec.lower() == "openaispec"):
                raise Exception(f"OpenAISpec only supported when streaming == True")
            elif (spec.lower() == "openaiembeddingspec"):
                spec_impl = OpenAIEmbeddingSpec()
            else:
                raise Exception(f"LitServe protocol spec {spec} not known")

        server = LitServer(
            SimpleLitAPI(
                checkpoint_dir=checkpoint_dir,
                quantize=quantize,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                devices=devices
                ),
            accelerator=accelerator,
            devices=1,  # We need to use the devices inside the `SimpleLitAPI` class
            spec=spec_impl
            )
        
    else:

        spec_impl = None
        if (spec != None):
            if (spec.lower() == "openaispec"):
                spec_impl = OpenAISpec()
            elif (spec.lower() == "openaiembeddingspec"):
                raise Exception(f"OpenAIEmbeddingSpec only supported when streaming == False")
            else:
                raise Exception(f"LitServe protocol spec {spec} not known")

        server = LitServer(
            OpenAIAPI(
                checkpoint_dir=checkpoint_dir,
                quantize=quantize,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                devices=devices  # We need to use the devices inside the `StreamLitAPI` class
                ),
            accelerator=accelerator,
            devices=1,
            stream=True,
            spec=spec_impl
            )

    server.run(port=port, generate_client_file=False)