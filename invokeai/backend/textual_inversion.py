"""Textual Inversion wrapper class."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from compel.embeddings_provider import BaseTextualInversionManager
from safetensors.torch import load_file
from transformers import CLIPTokenizer
from typing_extensions import Self
from .raw_model import RawModel

class TextualInversionModelRaw(RawModel):
    embedding: torch.Tensor  # [n, 768]|[n, 1280]
    embedding_2: Optional[torch.Tensor] = None  # [n, 768]|[n, 1280]   - for SDXL models

    @classmethod
    def from_checkpoint(
        cls,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        result = cls()  # TODO:

        if file_path.suffix == ".safetensors":
            state_dict = load_file(file_path.absolute().as_posix(), device="cpu")
        else:
            state_dict = torch.load(file_path, map_location="cpu")

        # both v1 and v2 format embeddings
        # difference mostly in metadata
        if "string_to_param" in state_dict:
            if len(state_dict["string_to_param"]) > 1:
                print(
                    f'Warn: Embedding "{file_path.name}" contains multiple tokens, which is not supported. The first',
                    " token will be used.",
                )

            result.embedding = next(iter(state_dict["string_to_param"].values()))

        # v3 (easynegative)
        elif "emb_params" in state_dict:
            result.embedding = state_dict["emb_params"]

        # v5(sdxl safetensors file)
        elif "clip_g" in state_dict and "clip_l" in state_dict:
            result.embedding = state_dict["clip_g"]
            result.embedding_2 = state_dict["clip_l"]

        # v4(diffusers bin files)
        else:
            result.embedding = next(iter(state_dict.values()))

            if len(result.embedding.shape) == 1:
                result.embedding = result.embedding.unsqueeze(0)

            if not isinstance(result.embedding, torch.Tensor):
                raise ValueError(f"Invalid embeddings file: {file_path.name}")

        return result


# no type hints for BaseTextualInversionManager?
class TextualInversionManager(BaseTextualInversionManager):  # type: ignore
    pad_tokens: Dict[int, List[int]]
    tokenizer: CLIPTokenizer

    def __init__(self, tokenizer: CLIPTokenizer):
        self.pad_tokens = {}
        self.tokenizer = tokenizer

    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: list[int]) -> list[int]:
        if len(self.pad_tokens) == 0:
            return token_ids

        if token_ids[0] == self.tokenizer.bos_token_id:
            raise ValueError("token_ids must not start with bos_token_id")
        if token_ids[-1] == self.tokenizer.eos_token_id:
            raise ValueError("token_ids must not end with eos_token_id")

        new_token_ids = []
        for token_id in token_ids:
            new_token_ids.append(token_id)
            if token_id in self.pad_tokens:
                new_token_ids.extend(self.pad_tokens[token_id])

        # Do not exceed the max model input size
        # The -2 here is compensating for compensate compel.embeddings_provider.get_token_ids(),
        # which first removes and then adds back the start and end tokens.
        max_length = list(self.tokenizer.max_model_input_sizes.values())[0] - 2
        if len(new_token_ids) > max_length:
            new_token_ids = new_token_ids[0:max_length]

        return new_token_ids
