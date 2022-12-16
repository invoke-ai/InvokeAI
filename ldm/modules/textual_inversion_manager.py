import os
from typing import Union

import torch
from attr import dataclass
from picklescan.scanner import scan_file_path

from ldm.invoke.concepts_lib import HuggingFaceConceptsLibrary
from ldm.modules.embedding_manager import get_clip_token_id_for_string
from ldm.modules.encoders.modules import FrozenCLIPEmbedder


@dataclass
class TextualInversion:
    trigger_string: str
    token_id: int
    embedding: torch.Tensor

    @property
    def embedding_vector_length(self) -> int:
        return self.embedding.shape[0]

class TextualInversionManager():
    def __init__(self, clip_embedder: FrozenCLIPEmbedder, full_precision: bool):
        self.clip_embedder = clip_embedder
        self.full_precision = full_precision
        self.hf_concepts_library = HuggingFaceConceptsLibrary()
        default_textual_inversions: list[TextualInversion] = []
        self.textual_inversions = default_textual_inversions

    def load_huggingface_concepts(self, concepts: list[str]):
        for concept_name in concepts:
            if concept_name in self.hf_concepts_library.concepts_loaded:
                continue
            bin_file = self.hf_concepts_library.get_concept_model_path(concept_name)
            if not bin_file:
                continue
            self.load_textual_inversion(bin_file)
            self.hf_concepts_library.concepts_loaded[concept_name]=True

    def get_all_trigger_strings(self) -> list[str]:
        return [ti.trigger_string for ti in self.textual_inversions]

    def load_textual_inversion(self, ckpt_path):
        scan_result = scan_file_path(ckpt_path)
        if scan_result.infected_files == 1:
            print(f'\n### Security Issues Found in Model: {scan_result.issues_count}')
            print('### For your safety, InvokeAI will not load this embed.')
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Handle .pt textual inversion files
        if 'string_to_token' in ckpt and 'string_to_param' in ckpt:
            filename = os.path.basename(ckpt_path)
            trigger_str = '.'.join(filename.split('.')[:-1]) # filename excluding extension
            if len(ckpt["string_to_token"]) > 1:
                print(f">> {ckpt_path} has >1 embedding, only the first will be used")

            string_to_param_dict = ckpt['string_to_param']
            embedding = list(string_to_param_dict.values())[0]
            self.add_textual_inversion(trigger_str, embedding)

        # Handle .bin textual inversion files from Huggingface Concepts
        # https://huggingface.co/sd-concepts-library
        else:
            for trigger_str in list(ckpt.keys()):
                embedding = ckpt[trigger_str]
                self.add_textual_inversion(trigger_str, embedding)

    def add_textual_inversion(self, trigger_str, embedding) -> int:
        """
        Add a textual inversion to be recognised.
        :param trigger_str: The trigger text in the prompt that activates this textual inversion. If unknown to the embedder's tokenizer, will be added.
        :param embedding: The actual embedding data that will be inserted into the conditioning at the point where the token_str appears.
        :return: The token id for the added embedding, either existing or newly-added.
        """
        if trigger_str in [ti.trigger_string for ti in self.textual_inversions]:
            print(f">> TextualInversionManager refusing to overwrite already-loaded token '{trigger_str}'")
            return
        if not self.full_precision:
            embedding = embedding.half()
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
        elif len(embedding.shape) > 2:
            raise ValueError(f"embedding shape {embedding.shape} is incorrect - must have shape [token_dim] or [V, token_dim] where V is vector length and token_dim is 768 for SD1 or 1280 for SD2")

        existing_token_id = get_clip_token_id_for_string(self.clip_embedder.tokenizer, trigger_str)
        if existing_token_id == self.clip_embedder.tokenizer.unk_token_id:
            num_tokens_added = self.clip_embedder.tokenizer.add_tokens(trigger_str)
            current_embeddings = self.clip_embedder.transformer.resize_token_embeddings(None)
            current_token_count = current_embeddings.num_embeddings
            new_token_count = current_token_count + num_tokens_added
            self.clip_embedder.transformer.resize_token_embeddings(new_token_count)

        token_id = get_clip_token_id_for_string(self.clip_embedder.tokenizer, trigger_str)
        self.textual_inversions.append(TextualInversion(
            trigger_string=trigger_str,
            token_id=token_id,
            embedding=embedding
        ))
        return token_id

    def has_textual_inversion_for_trigger_string(self, trigger_string: str) -> bool:
        try:
            ti = self.get_textual_inversion_for_trigger_string(trigger_string)
            return ti is not None
        except StopIteration:
            return False

    def get_textual_inversion_for_trigger_string(self, trigger_string: str) -> TextualInversion:
        return next(ti for ti in self.textual_inversions if ti.trigger_string == trigger_string)


    def get_textual_inversion_for_token_id(self, token_id: int) -> TextualInversion:
        return next(ti for ti in self.textual_inversions if ti.token_id == token_id)

    def expand_textual_inversion_token_ids(self, prompt_token_ids: list[int]) -> list[int]:
        """
        Insert padding tokens as necessary into the passed-in list of token ids to match any textual inversions it includes.

        :param prompt_token_ids: The prompt as a list of token ids (`int`s). Should not include bos and eos markers.
        :param pad_token_id: The token id to use to pad out the list to account for textual inversion vector lengths >1.
        :return: The prompt token ids with any necessary padding to account for textual inversions inserted. May be too
                long - caller is reponsible for truncating it if necessary and prepending/appending eos and bos token ids.
        """
        if len(prompt_token_ids) == 0:
            return prompt_token_ids

        if prompt_token_ids[0] == self.clip_embedder.tokenizer.bos_token_id:
            raise ValueError("prompt_token_ids must not start with bos_token_id")
        if prompt_token_ids[-1] == self.clip_embedder.tokenizer.eos_token_id:
            raise ValueError("prompt_token_ids must not end with eos_token_id")
        textual_inversion_token_ids = [ti.token_id for ti in self.textual_inversions]
        prompt_token_ids = prompt_token_ids[:]
        for i, token_id in reversed(list(enumerate(prompt_token_ids))):
            if token_id in textual_inversion_token_ids:
                textual_inversion = next(ti for ti in self.textual_inversions if ti.token_id == token_id)
                for pad_idx in range(1, textual_inversion.embedding_vector_length):
                    prompt_token_ids.insert(i+1, self.clip_embedder.tokenizer.pad_token_id)

        return prompt_token_ids

    def overwrite_textual_inversion_embeddings(self, prompt_token_ids: Union[torch.Tensor,list[int]], prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """
        For each token id in prompt_token_ids that refers to a loaded textual inversion, overwrite the corresponding
        row in `prompt_embeddings` with the textual inversion embedding. If the embedding has vector length >1, overwrite
        subsequent rows in `prompt_embeddings` as well.

        :param `prompt_token_ids`: Prompt token ids, already expanded to account for any textual inversions with vector length
            >1 (call `expand_textual_inversion_token_ids()` to do this), padded to max length, and including bos and eos markers.
        :param `prompt_embeddings`: Prompt embeddings tensor of shape with indices aligning to token ids in
            `prompt_token_ids` (i.e., also already expanded).
        :return: `The prompt_embeddings` tensor overwritten as appropriate with the textual inversion embeddings.
        """
        if type(prompt_token_ids) is torch.Tensor:
            if prompt_token_ids.shape != torch.Size([self.clip_embedder.max_length]):
                raise ValueError(f"prompt_token_ids must be a list of length {self.clip_embedder.max_length} or a tensor of shape [{self.clip_embedder.max_length}]")
            prompt_token_ids = list(prompt_token_ids.cpu().numpy())
        if prompt_embeddings.shape[0] != self.clip_embedder.max_length: # typically 77
            raise ValueError(f"prompt_embeddings must have {self.clip_embedder.max_length} entries (has: {prompt_embeddings.shape[0]})")
        if len(prompt_token_ids) > self.clip_embedder.max_length:
            raise ValueError(f"prompt_token_ids is too long (has {len(prompt_token_ids)} token ids, should have {self.clip_embedder.max_length})")
        if len(prompt_token_ids) < self.clip_embedder.max_length:
            raise ValueError(f"prompt_token_ids is too short (has {len(prompt_token_ids)} token ids, it must be fully padded out to {self.clip_embedder.max_length} entries)")
        if prompt_token_ids[0] != self.clip_embedder.tokenizer.bos_token_id or prompt_token_ids[-1] != self.clip_embedder.tokenizer.eos_token_id:
            raise ValueError("prompt_token_ids must start with with bos token id and end with the eos token id")

        textual_inversion_token_ids = [ti.token_id for ti in self.textual_inversions]
        pad_token_id = self.clip_embedder.tokenizer.pad_token_id
        overwritten_prompt_embeddings = prompt_embeddings.clone()
        for i, token_id in enumerate(prompt_token_ids):
            if token_id == pad_token_id:
                continue
            if token_id in textual_inversion_token_ids:
                textual_inversion = next(ti for ti in self.textual_inversions if ti.token_id == token_id)
                end_index = min(i + textual_inversion.embedding_vector_length, self.clip_embedder.max_length-1)
                count_to_overwrite = end_index - i
                for j in range(0, count_to_overwrite):
                    # only overwrite the textual inversion token id or the padding token id
                    if prompt_token_ids[i+j] != pad_token_id and prompt_token_ids[i+j] != token_id:
                        break
                    overwritten_prompt_embeddings[i+j] = textual_inversion.embedding[j]

        return overwritten_prompt_embeddings
