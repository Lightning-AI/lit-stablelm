import json
from pathlib import Path
from typing import Optional
import torch


class Tokenizer:
    def __init__(self, checkpoint_dir: Path) -> None:
        # some checkpoints have both files, `.model` takes precedence
        
        if (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.use_bos = self.bos_id is not None                    
            self.eos_id = self.processor.eos_id()
        elif (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                self.bos_id = config.get("bos_token_id")
                self.use_bos = self.bos_id  is not None
                self.eos_id = config.get("eos_token_id")

            elif (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.use_bos = any([config.get(check) for check in ['add_bos_token', 'add_prefix_space']])

                self.eos_id = self.token_to_id(config["eos_token"])             
            else:
                raise RuntimeError("Missing tokenizer config")     
                    
        else:
            raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError
        
        if bos is not None: 
            self.use_bos = bos

        if self.use_bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)
