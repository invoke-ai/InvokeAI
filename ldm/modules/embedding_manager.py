import os.path
from cmath import log
import torch
from attr import dataclass
from torch import nn

import sys

from ldm.invoke.concepts_lib import Concepts
from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
from picklescan.scanner import scan_file_path

PROGRESSIVE_SCALE = 2000


def get_clip_token_id_for_string(tokenizer, string):
    batch_encoding = tokenizer(
        string,
        truncation=True,
        max_length=77,
        return_length=True,
        return_overflowing_tokens=False,
        padding='max_length',
        return_tensors='pt',
    )
    tokens = batch_encoding['input_ids']
    assert (
        torch.count_nonzero(tokens - 49407) == 2
    ), f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    # assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]

@dataclass
class TextualInversion:
    token_string: str
    token_id: int
    embedding: torch.Tensor

    @property
    def embedding_vector_length(self) -> int:
        return self.embedding.shape[0]

class TextualInversionManager():
    def __init__(self, clip_embedder):
        self.clip_embedder = clip_embedder
        defatul_textual_inversions: list[TextualInversion] = []
        self.textual_inversions = defatul_textual_inversions

    def load_textual_inversion(self, ckpt_path, full_precision=True):

        scan_result = scan_file_path(ckpt_path)
        if scan_result.infected_files == 1:
            print(f'\n### Security Issues Found in Model: {scan_result.issues_count}')
            print('### For your safety, InvokeAI will not load this embed.')
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Handle .pt textual inversion files
        if 'string_to_token' in ckpt and 'string_to_param' in ckpt:
            filename = os.path.basename(ckpt_path)
            token_str = '.'.join(filename.split('.')[:-1]) # filename excluding extension
            if len(ckpt["string_to_token"]) > 1:
                print(f">> {ckpt_path} has >1 embedding, only the first will be used")

            string_to_param_dict = ckpt['string_to_param']
            embedding = list(string_to_param_dict.values())[0]
            self.add_textual_inversion(token_str, embedding, full_precision)

        # Handle .bin textual inversion files from Huggingface Concepts
        # https://huggingface.co/sd-concepts-library
        else:
            for token_str in list(ckpt.keys()):
                embedding = ckpt[token_str]
                self.add_textual_inversion(token_str, embedding, full_precision)

    def add_textual_inversion(self, token_str, embedding):
        """
        Add a textual inversion to be recognised.
        :param token_str: The trigger text in the prompt that activates this textual inversion. Should be unknown to the embedder's tokenizer.
        :param embedding: The actual embedding data that will be inserted into the conditioning at the point where the token_str appears.
        :return: The token id of the added embedding.
        """
        if token_str in [ti.token_string for ti in self.textual_inversions]:
            print(f">> Embedding manager refusing to overwrite already-loaded term '{token_str}'")
            return
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
        elif len(embedding.shape) > 2:
            raise ValueError(f"embedding shape {embedding.shape} is incorrect - must have shape [token_dim] or [V, token_dim] where V is vector length and token_dim is 768 for SD1 or 1280 for SD2")

        num_tokens_added = self.clip_embedder.tokenizer.add_tokens(token_str)
        current_embeddings = self.clip_embedder.transformer.resize_token_embeddings(None)
        current_token_count = current_embeddings.num_embeddings
        new_token_count = current_token_count + num_tokens_added
        self.clip_embedder.transformer.resize_token_embeddings(new_token_count)

        token_id = get_clip_token_id_for_string(self.clip_embedder.tokenizer, token_str)
        self.textual_inversions.append(TextualInversion(
            token_string=token_str,
            token_id=token_id,
            embedding=embedding
        ))

        return token_id

    def has_textual_inversion(self, token_str):
        return token_str in [ti.token_string for ti in self.textual_inversions]

    def expand_textual_inversion_token_ids(self, prompt_token_ids: list[int]) -> list[int]:
        """
        Insert padding tokens as necessary into the passed-in list of token ids to match any textual inversions it includes.

        :param prompt_token_ids: The prompt as a list of token ids (`int`s). Should not include bos and eos markers.
        :param pad_token_id: The token id to use to pad out the list to account for textual inversion vector lengths >1.
        :return: The prompt token ids with any necessary padding to account for textual inversions inserted. May be too
                long - caller is reponsible for truncating it if necessary and prepending/appending eos and bos token ids.
        """
        assert(prompt_token_ids[0] != self.clip_embedder.bos_token_id)
        assert(prompt_token_ids[-1] != self.clip_embedder.eos_token_id)
        textual_inversion_token_ids = [ti.token_id for ti in self.textual_inversions]
        for i, token_id in reversed(list(enumerate(prompt_token_ids))):
            if token_id in textual_inversion_token_ids:
                textual_inversion = next(ti for ti in self.textual_inversions if ti.token_id == token_id)
                for pad_idx in range(1, textual_inversion.embedding_vector_length):
                    prompt_token_ids.insert(i+1, self.clip_embedder.pad_token_id)

        return prompt_token_ids

    def overwrite_textual_inversion_embeddings(self, prompt_token_ids: list[int], prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """
        For each token id in prompt_token_ids that refers to a loaded textual inversion, overwrite the corresponding
        row in `prompt_embeddings` with the textual inversion embedding. If the embedding has vector length >1, overwrite
        subsequent rows in `prompt_embeddings` as well.

        :param `prompt_token_ids`: Prompt token ids, already expanded to account for any textual inversions with vector lenght
            >1 (call `expand_textual_inversion_token_ids()` to do this)
        :param `prompt_embeddings`: Prompt embeddings tensor of shape with indices aligning to token ids in
            `prompt_token_ids` (i.e., also already expanded).
        :return: `The prompt_embeddings` tensor overwritten as appropriate with the textual inversion embeddings.
        """
        assert(prompt_embeddings.shape[0] == self.clip_embedder.max_length, f"prompt_embeddings must have 77 entries (has: {prompt_embeddings.shape[0]})")
        textual_inversion_token_ids = [ti.token_id for ti in self.textual_inversions]
        for i, token_id in enumerate(prompt_token_ids):
            if token_id == pad_token_id:
                continue
            if token_id in textual_inversion_token_ids:
                textual_inversion = next(ti for ti in self.textual_inversions if ti.token_id == token_id)
                for j in range(0, textual_inversion.embedding_vector_length):
                    # only overwrite the textual inversion token id or the padding token id
                    if prompt_token_ids[i+j] != self.clip_embedder.pad_token_id and prompt_token_ids[i+j] != token_id:
                        break
                    prompt_embeddings[i+j] = textual_inversion.embedding[j]

        return prompt_embeddings





class EmbeddingManager(nn.Module):
    def __init__(
        self,
        embedder,
        placeholder_strings=None,
        initializer_words=None,
        per_image_tokens=False,
        num_vectors_per_token=1,
        progressive_words=False,
        **kwargs,
    ):
        super().__init__()

        self.embedder = embedder
        self.concepts_library=Concepts()
        self.concepts_loaded = dict()

        self.string_to_token_dict = {}
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = (
            nn.ParameterDict()
        )   # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(
            embedder, 'tokenizer'
        ):   # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(
                get_clip_token_id_for_string, embedder.tokenizer
            )
            get_embedding_for_tkn = partial(
                get_embedding_for_clip_token,
                embedder.transformer.text_model.embeddings,
            )
            # per bug report #572
            #token_dim = 1280
            token_dim = 768
        else:   # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(
                get_bert_token_for_string, embedder.tknz_fn
            )
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):

            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(
                        init_word_token.cpu()
                    )

                token_params = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(
                        num_vectors_per_token, 1
                    ),
                    requires_grad=True,
                )
                self.initial_embeddings[
                    placeholder_string
                ] = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(
                        num_vectors_per_token, 1
                    ),
                    requires_grad=False,
                )
            else:
                token_params = torch.nn.Parameter(
                    torch.rand(
                        size=(num_vectors_per_token, token_dim),
                        requires_grad=True,
                    )
                )

            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(
        self,
        tokenized_text,
        embedded_text,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        for (
            placeholder_string,
            placeholder_token,
        ) in self.string_to_token_dict.items():

            placeholder_embedding = self.string_to_param_dict[
                placeholder_string
            ].to(device)

            if self.progressive_words:
                self.progressive_counter += 1
                max_step_tokens = (
                    1 + self.progressive_counter // PROGRESSIVE_SCALE
                )
            else:
                max_step_tokens = self.max_vectors_per_token

            num_vectors_for_token = min(
                placeholder_embedding.shape[0], max_step_tokens
            )

            placeholder_rows, placeholder_cols = torch.where(
                tokenized_text == placeholder_token.to(tokenized_text.device)
            )

            if placeholder_rows.nelement() == 0:
                continue

            sorted_cols, sort_idx = torch.sort(
                placeholder_cols, descending=True
            )
            sorted_rows = placeholder_rows[sort_idx]

            for idx in range(sorted_rows.shape[0]):
                row = sorted_rows[idx]
                col = sorted_cols[idx]

                new_token_row = torch.cat(
                    [
                        tokenized_text[row][:col],
                        placeholder_token.repeat(num_vectors_for_token).to(
                            device
                        ),
                        tokenized_text[row][col + 1 :],
                    ],
                    axis=0,
                )[:n]
                new_embed_row = torch.cat(
                    [
                        embedded_text[row][:col],
                        placeholder_embedding[:num_vectors_for_token],
                        embedded_text[row][col + 1 :],
                    ],
                    axis=0,
                )[:n]

                embedded_text[row] = new_embed_row
                tokenized_text[row] = new_token_row

        return embedded_text

    def save(self, ckpt_path):
        torch.save(
            {
                'string_to_token': self.string_to_token_dict,
                'string_to_param': self.string_to_param_dict,
            },
            ckpt_path,
        )

    def load_concepts(self, concepts:list[str], full=True):
        bin_files = list()
        for concept_name in concepts:
            if concept_name in self.concepts_loaded:
                continue
            else:
                bin_file = self.concepts_library.get_concept_model_path(concept_name)
                if not bin_file:
                    continue
                bin_files.append(bin_file)
                self.concepts_loaded[concept_name]=True
        self.load(bin_files, full)

    def list_terms(self) -> list[str]:
        return self.concepts_loaded.keys()

    def load(self, ckpt_paths, full=True):
        if len(ckpt_paths) == 0:
            return
        if type(ckpt_paths) != list:
            ckpt_paths = [ckpt_paths]
        ckpt_paths = self._expand_directories(ckpt_paths)
        for c in ckpt_paths:
            self._load(c,full)
        # remember that we know this term and don't try to download it again from the concepts library
        # note that if the concept name is also provided and different from the trigger term, they
        # both will be stored in this dictionary
        for term in self.string_to_param_dict.keys():
            term = term.strip('<').strip('>')
            self.concepts_loaded[term] = True
        print(f'>> Current embedding manager terms: {", ".join(self.string_to_param_dict.keys())}')

    def _expand_directories(self, paths:list[str]):
        expanded_paths = list()
        for path in paths:
            if os.path.isfile(path):
                expanded_paths.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for name in files:
                        expanded_paths.append(os.path.join(root,name))
        return [x for x in expanded_paths if os.path.splitext(x)[1] in ('.pt','.bin')]


    def get_embedding_norms_squared(self):
        all_params = torch.cat(
            list(self.string_to_param_dict.values()), axis=0
        )   # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(
            axis=-1
        )              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):

        loss = 0.0
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = (
                loss
                + (optimized - coarse)
                @ (optimized - coarse).T
                / num_embeddings
            )

        return loss
