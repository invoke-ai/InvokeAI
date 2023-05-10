import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

import safetensors.torch
import torch

from compel.embeddings_provider import BaseTextualInversionManager
from picklescan.scanner import scan_file_path
from transformers import CLIPTextModel, CLIPTokenizer

import invokeai.backend.util.logging as logger
from .concepts_lib import HuggingFaceConceptsLibrary

@dataclass
class EmbeddingInfo:
    name: str
    embedding: torch.Tensor
    num_vectors_per_token: int
    token_dim: int 
    trained_steps: int = None
    trained_model_name: str = None
    trained_model_checksum: str = None

@dataclass
class TextualInversion:
    trigger_string: str
    embedding: torch.Tensor
    trigger_token_id: Optional[int] = None
    pad_token_ids: Optional[list[int]] = None

    @property
    def embedding_vector_length(self) -> int:
        return self.embedding.shape[0]


class TextualInversionManager(BaseTextualInversionManager):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        full_precision: bool = True,
    ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.full_precision = full_precision
        self.hf_concepts_library = HuggingFaceConceptsLibrary()
        self.trigger_to_sourcefile = dict()
        default_textual_inversions: list[TextualInversion] = []
        self.textual_inversions = default_textual_inversions

    def load_huggingface_concepts(self, concepts: list[str]):
        for concept_name in concepts:
            if concept_name in self.hf_concepts_library.concepts_loaded:
                continue
            trigger = self.hf_concepts_library.concept_to_trigger(concept_name)
            if (
                self.has_textual_inversion_for_trigger_string(trigger)
                or self.has_textual_inversion_for_trigger_string(concept_name)
                or self.has_textual_inversion_for_trigger_string(f"<{concept_name}>")
            ):  # in case a token with literal angle brackets encountered
                logger.info(f"Loaded local embedding for trigger {concept_name}")
                continue
            bin_file = self.hf_concepts_library.get_concept_model_path(concept_name)
            if not bin_file:
                continue
            logger.info(f"Loaded remote embedding for trigger {concept_name}")
            self.load_textual_inversion(bin_file)
            self.hf_concepts_library.concepts_loaded[concept_name] = True

    def get_all_trigger_strings(self) -> list[str]:
        return [ti.trigger_string for ti in self.textual_inversions]

    def load_textual_inversion(
        self, ckpt_path: Union[str, Path], defer_injecting_tokens: bool = False
    ):
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.is_file():
            return

        if str(ckpt_path).endswith(".DS_Store"):
            return

        embedding_list = self._parse_embedding(str(ckpt_path))
        for embedding_info in embedding_list:
            if (self.text_encoder.get_input_embeddings().weight.data[0].shape[0] != embedding_info.token_dim):
                logger.warning(
                    f"Notice: {ckpt_path.parents[0].name}/{ckpt_path.name} was trained on a model with an incompatible token dimension: {self.text_encoder.get_input_embeddings().weight.data[0].shape[0]} vs {embedding_info.token_dim}."
                )
                continue

            # Resolve the situation in which an earlier embedding has claimed the same
            # trigger string. We replace the trigger with '<source_file>', as we used to.
            trigger_str = embedding_info.name
            sourcefile = (
                f"{ckpt_path.parent.name}/{ckpt_path.name}"
                if ckpt_path.name == "learned_embeds.bin"
                else ckpt_path.name
            )

            if trigger_str in self.trigger_to_sourcefile:
                replacement_trigger_str = (
                    f"<{ckpt_path.parent.name}>"
                    if ckpt_path.name == "learned_embeds.bin"
                    else f"<{ckpt_path.stem}>"
                )
                logger.info(
                    f"{sourcefile}: Trigger token '{trigger_str}' is already claimed by '{self.trigger_to_sourcefile[trigger_str]}'. Trigger this concept with {replacement_trigger_str}"
                )
                trigger_str = replacement_trigger_str

            try:
                self._add_textual_inversion(
                    trigger_str,
                    embedding_info.embedding,
                    defer_injecting_tokens=defer_injecting_tokens,
                )
                # remember which source file claims this trigger
                self.trigger_to_sourcefile[trigger_str] = sourcefile

            except ValueError as e:
                logger.debug(f'Ignoring incompatible embedding {embedding_info["name"]}')
                logger.debug(f"The error was {str(e)}")

    def _add_textual_inversion(
        self, trigger_str, embedding, defer_injecting_tokens=False
    ) -> Optional[TextualInversion]:
        """
        Add a textual inversion to be recognised.
        :param trigger_str: The trigger text in the prompt that activates this textual inversion. If unknown to the embedder's tokenizer, will be added.
        :param embedding: The actual embedding data that will be inserted into the conditioning at the point where the token_str appears.
        :return: The token id for the added embedding, either existing or newly-added.
        """
        if trigger_str in [ti.trigger_string for ti in self.textual_inversions]:
            logger.warning(
                f"TextualInversionManager refusing to overwrite already-loaded token '{trigger_str}'"
            )
            return
        if not self.full_precision:
            embedding = embedding.half()
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
        elif len(embedding.shape) > 2:
            raise ValueError(
                f"** TextualInversionManager cannot add {trigger_str} because the embedding shape {embedding.shape} is incorrect. The embedding must have shape [token_dim] or [V, token_dim] where V is vector length and token_dim is 768 for SD1 or 1280 for SD2."
            )

        try:
            ti = TextualInversion(trigger_string=trigger_str, embedding=embedding)
            if not defer_injecting_tokens:
                self._inject_tokens_and_assign_embeddings(ti)
            self.textual_inversions.append(ti)
            return ti

        except ValueError as e:
            if str(e).startswith("Warning"):
                logger.warning(f"{str(e)}")
            else:
                traceback.print_exc()
                logger.error(
                    f"TextualInversionManager was unable to add a textual inversion with trigger string {trigger_str}."
                )
                raise

    def _inject_tokens_and_assign_embeddings(self, ti: TextualInversion) -> int:
        if ti.trigger_token_id is not None:
            raise ValueError(
                f"Tokens already injected for textual inversion with trigger '{ti.trigger_string}'"
            )

        trigger_token_id = self._get_or_create_token_id_and_assign_embedding(
            ti.trigger_string, ti.embedding[0]
        )

        if ti.embedding_vector_length > 1:
            # for embeddings with vector length > 1
            pad_token_strings = [
                ti.trigger_string + "-!pad-" + str(pad_index)
                for pad_index in range(1, ti.embedding_vector_length)
            ]
            # todo: batched UI for faster loading when vector length >2
            pad_token_ids = [
                self._get_or_create_token_id_and_assign_embedding(
                    pad_token_str, ti.embedding[1 + i]
                )
                for (i, pad_token_str) in enumerate(pad_token_strings)
            ]
        else:
            pad_token_ids = []

        ti.trigger_token_id = trigger_token_id
        ti.pad_token_ids = pad_token_ids
        return ti.trigger_token_id

    def has_textual_inversion_for_trigger_string(self, trigger_string: str) -> bool:
        try:
            ti = self.get_textual_inversion_for_trigger_string(trigger_string)
            return ti is not None
        except StopIteration:
            return False

    def get_textual_inversion_for_trigger_string(
        self, trigger_string: str
    ) -> TextualInversion:
        return next(
            ti for ti in self.textual_inversions if ti.trigger_string == trigger_string
        )

    def get_textual_inversion_for_token_id(self, token_id: int) -> TextualInversion:
        return next(
            ti for ti in self.textual_inversions if ti.trigger_token_id == token_id
        )

    def create_deferred_token_ids_for_any_trigger_terms(
        self, prompt_string: str
    ) -> list[int]:
        injected_token_ids = []
        for ti in self.textual_inversions:
            if ti.trigger_token_id is None and ti.trigger_string in prompt_string:
                if ti.embedding_vector_length > 1:
                    logger.info(
                        f"Preparing tokens for textual inversion {ti.trigger_string}..."
                    )
                try:
                    self._inject_tokens_and_assign_embeddings(ti)
                except ValueError as e:
                    logger.debug(
                        f"Ignoring incompatible embedding trigger {ti.trigger_string}"
                    )
                    logger.debug(f"The error was {str(e)}")
                    continue
                injected_token_ids.append(ti.trigger_token_id)
                injected_token_ids.extend(ti.pad_token_ids)
        return injected_token_ids

    def expand_textual_inversion_token_ids_if_necessary(
        self, prompt_token_ids: list[int]
    ) -> list[int]:
        """
        Insert padding tokens as necessary into the passed-in list of token ids to match any textual inversions it includes.

        :param prompt_token_ids: The prompt as a list of token ids (`int`s). Should not include bos and eos markers.
        :return: The prompt token ids with any necessary padding to account for textual inversions inserted. May be too
                long - caller is responsible for prepending/appending eos and bos token ids, and truncating if necessary.
        """
        if len(prompt_token_ids) == 0:
            return prompt_token_ids

        if prompt_token_ids[0] == self.tokenizer.bos_token_id:
            raise ValueError("prompt_token_ids must not start with bos_token_id")
        if prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            raise ValueError("prompt_token_ids must not end with eos_token_id")
        textual_inversion_trigger_token_ids = [
            ti.trigger_token_id for ti in self.textual_inversions
        ]
        prompt_token_ids = prompt_token_ids.copy()
        for i, token_id in reversed(list(enumerate(prompt_token_ids))):
            if token_id in textual_inversion_trigger_token_ids:
                textual_inversion = next(
                    ti
                    for ti in self.textual_inversions
                    if ti.trigger_token_id == token_id
                )
                for pad_idx in range(0, textual_inversion.embedding_vector_length - 1):
                    prompt_token_ids.insert(
                        i + pad_idx + 1, textual_inversion.pad_token_ids[pad_idx]
                    )

        return prompt_token_ids

    def _get_or_create_token_id_and_assign_embedding(
        self, token_str: str, embedding: torch.Tensor
    ) -> int:
        if len(embedding.shape) != 1:
            raise ValueError(
                "Embedding has incorrect shape - must be [token_dim] where token_dim is 768 for SD1 or 1280 for SD2"
            )
        existing_token_id = self.tokenizer.convert_tokens_to_ids(token_str)
        if existing_token_id == self.tokenizer.unk_token_id:
            num_tokens_added = self.tokenizer.add_tokens(token_str)
            current_embeddings = self.text_encoder.resize_token_embeddings(None)
            current_token_count = current_embeddings.num_embeddings
            new_token_count = current_token_count + num_tokens_added
            # the following call is slow - todo make batched for better performance with vector length >1
            self.text_encoder.resize_token_embeddings(new_token_count)

        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
        if token_id == self.tokenizer.unk_token_id:
            raise RuntimeError(f"Unable to find token id for token '{token_str}'")
        if (
            self.text_encoder.get_input_embeddings().weight.data[token_id].shape
            != embedding.shape
        ):
            raise ValueError(
                f"Warning. Cannot load embedding for {token_str}. It was trained on a model with token dimension {embedding.shape[0]}, but the current model has token dimension {self.text_encoder.get_input_embeddings().weight.data[token_id].shape[0]}."
            )
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

        return token_id


    def _parse_embedding(self, embedding_file: str)->List[EmbeddingInfo]:
        suffix = Path(embedding_file).suffix
        try:
            if suffix in [".pt",".ckpt",".bin"]:
                scan_result = scan_file_path(embedding_file)
                if scan_result.infected_files > 0:
                    logger.critical(
                        f"Security Issues Found in Model: {scan_result.issues_count}"
                    )
                    logger.critical("For your safety, InvokeAI will not load this embed.")
                    return list()
                ckpt = torch.load(embedding_file,map_location="cpu")
            else:
                ckpt = safetensors.torch.load_file(embedding_file)
        except Exception as e:
            logger.warning(f"Notice: unrecognized embedding file format: {embedding_file}: {e}")
            return list()
        
        # try to figure out what kind of embedding file it is and parse accordingly
        keys = list(ckpt.keys())
        if all(x in keys for x in ['string_to_token','string_to_param','name','step']):
            return self._parse_embedding_v1(ckpt, embedding_file)     # example rem_rezero.pt
        
        elif all(x in keys for x in ['string_to_token','string_to_param']):
            return self._parse_embedding_v2(ckpt, embedding_file)     # example midj-strong.pt
        
        elif 'emb_params' in keys:
            return self._parse_embedding_v3(ckpt, embedding_file)     # example easynegative.safetensors
        
        else:
            return self._parse_embedding_v4(ckpt, embedding_file)     # usually a '.bin' file

    def _parse_embedding_v1(self, embedding_ckpt: dict, file_path: str)->List[EmbeddingInfo]:
        basename = Path(file_path).stem
        logger.debug(f'Loading v1 embedding file: {basename}')

        embeddings = list()
        token_counter = -1
        for token,embedding in embedding_ckpt["string_to_param"].items():
            if token_counter < 0:
                trigger = embedding_ckpt["name"]
            elif token_counter == 0:
                trigger = '<basename>'
            else:
                trigger = f'<{basename}-{int(token_counter:=token_counter)}>'
            token_counter += 1
            embedding_info = EmbeddingInfo(
                name = trigger,
                embedding = embedding,
                num_vectors_per_token = embedding.size()[0],
                token_dim = embedding.size()[1],
                trained_steps = embedding_ckpt["step"],
                trained_model_name = embedding_ckpt["sd_checkpoint_name"],
                trained_model_checksum = embedding_ckpt["sd_checkpoint"]
            )
            embeddings.append(embedding_info)
        return embeddings

    def _parse_embedding_v2 (
        self, embedding_ckpt: dict, file_path: str
    ) -> List[EmbeddingInfo]:
        """
        This handles embedding .pt file variant #2.
        """
        basename = Path(file_path).stem
        logger.debug(f'Loading v2 embedding file: {basename}')
        embeddings = list()
        
        if isinstance(
            list(embedding_ckpt["string_to_token"].values())[0], torch.Tensor
        ):
            token_counter = 0
            for token,embedding in embedding_ckpt["string_to_param"].items():
                trigger = token if token != '*' \
                    else f'<{basename}>' if token_counter == 0 \
                         else f'<{basename}-{int(token_counter:=token_counter+1)}>'
                embedding_info = EmbeddingInfo(
                    name = trigger,
                    embedding = embedding,
                    num_vectors_per_token = embedding.size()[0],
                    token_dim = embedding.size()[1],
                )
                embeddings.append(embedding_info)
        else:
            logger.warning(f"{basename}: Unrecognized embedding format")

        return embeddings

    def _parse_embedding_v3(self, embedding_ckpt: dict, file_path: str)->List[EmbeddingInfo]:
        """
        Parse 'version 3' of the .pt textual inversion embedding files.
        """
        basename = Path(file_path).stem
        logger.debug(f'Loading v3 embedding file: {basename}')
        embedding = embedding_ckpt['emb_params']
        embedding_info = EmbeddingInfo(
            name = f'<{basename}>',
            embedding = embedding,
            num_vectors_per_token = embedding.size()[0],
            token_dim = embedding.size()[1],
        )
        return [embedding_info]
    
    def _parse_embedding_v4(self, embedding_ckpt: dict, filepath: str)->List[EmbeddingInfo]:
        """
        Parse 'version 4' of the textual inversion embedding files. This one
        is usually associated with .bin files trained by HuggingFace diffusers.
        """
        basename = Path(filepath).stem
        short_path = Path(filepath).parents[0].name+'/'+Path(filepath).name
        
        logger.debug(f'Loading v4 embedding file: {short_path}')
        
        embeddings = list()
        if list(embedding_ckpt.keys()) == 0:
            logger.warning(f"Invalid embeddings file: {short_path}")
        else:
            for token,embedding in embedding_ckpt.items():
                embedding_info = EmbeddingInfo(
                    name = token or f"<{basename}>",
                    embedding = embedding,
                    num_vectors_per_token = 1,  # All Concepts seem to default to 1
                    token_dim = embedding.size()[0],
                )
                embeddings.append(embedding_info)
        return embeddings
