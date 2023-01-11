import os.path
from cmath import log
import torch
from attr import dataclass
from torch import nn

import sys

from ldm.invoke.concepts_lib import HuggingFaceConceptsLibrary
from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
from picklescan.scanner import scan_file_path

PROGRESSIVE_SCALE = 2000


def get_clip_token_id_for_string(tokenizer: CLIPTokenizer, token_str: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    return token_id

def get_embedding_for_clip_token_id(embedder, token_id):
    if type(token_id) is not torch.Tensor:
        token_id = torch.tensor(token_id, dtype=torch.int)
    return embedder(token_id.unsqueeze(0))[0, 0]


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
        self.concepts_library=HuggingFaceConceptsLibrary()

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
            get_token_id_for_string = partial(
                get_clip_token_id_for_string, embedder.tokenizer
            )
            get_embedding_for_tkn_id = partial(
                get_embedding_for_clip_token_id,
                embedder.transformer.text_model.embeddings,
            )
            # per bug report #572
            #token_dim = 1280
            token_dim = 768
        else:   # using LDM's BERT encoder
            self.is_clip = False
            get_token_id_for_string = partial(
                get_bert_token_id_for_string, embedder.tknz_fn
            )
            get_embedding_for_tkn_id = embedder.transformer.token_emb
            token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):

            token_id = get_token_id_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token_id = get_token_id_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn_id(init_word_token_id)

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

            self.string_to_token_dict[placeholder_string] = token_id
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(
        self,
        tokenized_text,
        embedded_text,
    ):
        # torch.save(embedded_text, '/tmp/embedding-manager-uglysonic-pre-rewrite.pt')

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
                tokenized_text == placeholder_token
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
                        torch.tensor([placeholder_token] * num_vectors_for_token, device=device),
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

    def _load(self, ckpt_path, full=True):
        try:
            scan_result = scan_file_path(ckpt_path)
            if scan_result.infected_files == 1:
                print(f'\n### Security Issues Found in Model: {scan_result.issues_count}')
                print('### For your safety, InvokeAI will not load this embed.')
                return
        except Exception:
            print(f"### WARNING::: Invalid or corrupt embeddings found. Ignoring: {ckpt_path}")
            return

        embedding_info = self.parse_embedding(ckpt_path)
        if embedding_info:
            self.max_vectors_per_token = embedding_info['num_vectors_per_token']
            self.add_embedding(embedding_info['name'], embedding_info['embedding'], full)
        else:
            print(f'>> Failed to load embedding located at {ckpt_path}. Unsupported file.')

    def add_embedding(self, token_str, embedding, full):
        if token_str in self.string_to_param_dict:
            print(f">> Embedding manager refusing to overwrite already-loaded term '{token_str}'")
            return
        if not full:
            embedding = embedding.half()
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)

        existing_token_id = get_clip_token_id_for_string(self.embedder.tokenizer, token_str)
        if existing_token_id == self.embedder.tokenizer.unk_token_id:
            num_tokens_added = self.embedder.tokenizer.add_tokens(token_str)
            current_embeddings = self.embedder.transformer.resize_token_embeddings(None)
            current_token_count = current_embeddings.num_embeddings
            new_token_count = current_token_count + num_tokens_added
            self.embedder.transformer.resize_token_embeddings(new_token_count)

        token_id = get_clip_token_id_for_string(self.embedder.tokenizer, token_str)
        self.string_to_token_dict[token_str] = token_id
        self.string_to_param_dict[token_str] = torch.nn.Parameter(embedding)

    def parse_embedding(self, embedding_file: str):
        file_type = embedding_file.split('.')[-1]
        if file_type == 'pt':
            return self.parse_embedding_pt(embedding_file)
        elif file_type == 'bin':
            return self.parse_embedding_bin(embedding_file)
        else:
            print(f'>> Not a recognized embedding file: {embedding_file}')

    def parse_embedding_pt(self, embedding_file):
        embedding_ckpt = torch.load(embedding_file, map_location='cpu')
        embedding_info = {}

        # Check if valid embedding file
        if 'string_to_token' and 'string_to_param' in embedding_ckpt:

            # Catch variants that do not have the expected keys or values.
            try:
                embedding_info['name'] = embedding_ckpt['name'] or os.path.basename(os.path.splitext(embedding_file)[0])

                # Check num of embeddings and warn user only the first will be used
                embedding_info['num_of_embeddings'] = len(embedding_ckpt["string_to_token"])
                if embedding_info['num_of_embeddings'] > 1:
                    print('>> More than 1 embedding found. Will use the first one')

                embedding = list(embedding_ckpt['string_to_param'].values())[0]
            except (AttributeError,KeyError):
                return self.handle_broken_pt_variants(embedding_ckpt, embedding_file)

            embedding_info['embedding'] = embedding
            embedding_info['num_vectors_per_token'] = embedding.size()[0]
            embedding_info['token_dim'] = embedding.size()[1]

            try:
                embedding_info['trained_steps'] = embedding_ckpt['step']
                embedding_info['trained_model_name'] = embedding_ckpt['sd_checkpoint_name']
                embedding_info['trained_model_checksum'] = embedding_ckpt['sd_checkpoint']
            except AttributeError:
                print(">> No Training Details Found. Passing ...")

        # .pt files found at https://cyberes.github.io/stable-diffusion-textual-inversion-models/
        # They are actually .bin files
        elif len(embedding_ckpt.keys())==1:
            print('>> Detected .bin file masquerading as .pt file')
            embedding_info = self.parse_embedding_bin(embedding_file)

        else:
            print('>> Invalid embedding format')
            embedding_info = None

        return embedding_info

    def parse_embedding_bin(self, embedding_file):
        embedding_ckpt = torch.load(embedding_file, map_location='cpu')
        embedding_info = {}

        if list(embedding_ckpt.keys()) == 0:
            print(">> Invalid concepts file")
            embedding_info = None
        else:
            for token in list(embedding_ckpt.keys()):
                embedding_info['name'] = token or os.path.basename(os.path.splitext(embedding_file)[0])
                embedding_info['embedding'] = embedding_ckpt[token]
                embedding_info['num_vectors_per_token'] = 1 # All Concepts seem to default to 1
                embedding_info['token_dim'] = embedding_info['embedding'].size()[0]

        return embedding_info

    def handle_broken_pt_variants(self, embedding_ckpt:dict, embedding_file:str)->dict:
        '''
        This handles the broken .pt file variants. We only know of one at present.
        '''
        embedding_info = {}
        if isinstance(list(embedding_ckpt['string_to_token'].values())[0],torch.Tensor):
            print(f'>> Variant Embedding Detected. Parsing: {embedding_file}') # example at https://github.com/invoke-ai/InvokeAI/issues/1829
            token = list(embedding_ckpt['string_to_token'].keys())[0]
            embedding_info['name'] = os.path.basename(os.path.splitext(embedding_file)[0])
            embedding_info['embedding'] = embedding_ckpt['string_to_param'].state_dict()[token]
            embedding_info['num_vectors_per_token'] = embedding_info['embedding'].shape[0]
            embedding_info['token_dim'] = embedding_info['embedding'].size()[0]
        else:
            print('>> Invalid embedding format')
            embedding_info = None

        return embedding_info

    def has_embedding_for_token(self, token_str):
        return token_str in self.string_to_token_dict

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
