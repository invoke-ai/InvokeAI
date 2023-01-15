import math
import os.path
from typing import Optional

import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia
from ldm.invoke.devices import choose_torch_device
from ldm.invoke.globals import Globals, global_cache_dir
#from ldm.modules.textual_inversion_manager import TextualInversionManager

from ldm.modules.x_transformer import (
    Encoder,
    TransformerWrapper,
)  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


def _expand_mask(mask, dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = (
        mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    )

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size,
        max_seq_len=77,
        device=choose_torch_device(),
    ):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(
            self, device=choose_torch_device(), vq_interface=True, max_length=77
    ):
        super().__init__()
        from transformers import (
            BertTokenizerFast,
        )

        cache = global_cache_dir('hub')
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                'bert-base-uncased',
                cache_dir=cache,
                local_files_only=True
            )
        except OSError:
            raise SystemExit(
                "* Couldn't load Bert tokenizer files. Try running scripts/preload_models.py from an internet-conected machine."
            )
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        tokens = batch_encoding['input_ids'].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
            self,
            n_embed,
            n_layer,
            vocab_size=30522,
            max_seq_len=77,
            device=choose_torch_device(),
            use_tokenizer=True,
            embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(
                vq_interface=False, max_length=max_seq_len
            )
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text, embedding_manager=None):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(
            tokens, return_embeddings=True, embedding_manager=embedding_manager
        )
        return z

    def encode(self, text, **kwargs):
        # output of length 77
        return self(text, **kwargs)


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method='bilinear',
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            'nearest',
            'linear',
            'bilinear',
            'trilinear',
            'bicubic',
            'area',
        ]
        self.multiplier = multiplier
        self.interpolator = partial(
            torch.nn.functional.interpolate, mode=method
        )
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.'
            )
            self.channel_mapper = nn.Conv2d(
                in_channels, out_channels, 1, bias=bias
            )

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    tokenizer: CLIPTokenizer
    transformer: CLIPTextModel

    def __init__(
        self,
        version:str='openai/clip-vit-large-patch14',
        max_length:int=77,
        tokenizer:Optional[CLIPTokenizer]=None,
        transformer:Optional[CLIPTextModel]=None,
    ):
        super().__init__()
        cache = global_cache_dir('hub')
        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained(
            version,
            cache_dir=cache,
            local_files_only=True
        )
        self.transformer = transformer or CLIPTextModel.from_pretrained(
            version,
            cache_dir=cache,
            local_files_only=True
        )
        self.max_length = max_length
        self.freeze()

        def embedding_forward(
            self,
            input_ids=None,
            position_ids=None,
            inputs_embeds=None,
            embedding_manager=None,
        ) -> torch.Tensor:

            seq_length = (
                input_ids.shape[-1]
                if input_ids is not None
                else inputs_embeds.shape[-2]
            )

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]

            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)

            if embedding_manager is not None:
                inputs_embeds = embedding_manager(input_ids, inputs_embeds)

            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings

            return embeddings

        self.transformer.text_model.embeddings.forward = (
            embedding_forward.__get__(self.transformer.text_model.embeddings)
        )

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict
                if return_dict is not None
                else self.config.use_return_dict
            )

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(
            self.transformer.text_model.encoder
        )

        def text_encoder_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict
                if return_dict is not None
                else self.config.use_return_dict
            )

            if input_ids is None:
                raise ValueError('You have to specify either input_ids')

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                embedding_manager=embedding_manager,
            )

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(
                bsz, seq_len, hidden_states.dtype
            ).to(hidden_states.device)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(
                    attention_mask, hidden_states.dtype
                )

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = self.final_layer_norm(last_hidden_state)

            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(
            self.transformer.text_model
        )

        def transformer_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager=embedding_manager,
            )

        self.transformer.forward = transformer_forward.__get__(
            self.transformer
        )

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, **kwargs):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        tokens = batch_encoding['input_ids'].to(self.device)
        z = self.transformer(input_ids=tokens, **kwargs)

        return z

    def encode(self, text, **kwargs):
        return self(text, **kwargs)

    @property
    def device(self):
        return self.transformer.device

    @device.setter
    def device(self, device):
        self.transformer.to(device=device)

class WeightedFrozenCLIPEmbedder(FrozenCLIPEmbedder):

    fragment_weights_key = "fragment_weights"
    return_tokens_key = "return_tokens"

    def set_textual_inversion_manager(self, manager): #TextualInversionManager):
        # TODO all of the weighting and expanding stuff needs be moved out of this class
        self.textual_inversion_manager = manager

    def forward(self, text: list, **kwargs):
        # TODO all of the weighting and expanding stuff needs be moved out of this class
        '''

        :param text: A batch of prompt strings, or, a batch of lists of fragments of prompt strings to which different
        weights shall be applied.
        :param kwargs: If the keyword arg "fragment_weights" is passed, it shall contain a batch of lists of weights
        for the prompt fragments. In this case text must contain batches of lists of prompt fragments.
        :return: A tensor of shape (B, 77, 768) containing weighted embeddings
        '''
        if self.fragment_weights_key not in kwargs:
            # fallback to base class implementation
            return super().forward(text, **kwargs)

        fragment_weights = kwargs[self.fragment_weights_key]
        # self.transformer doesn't like receiving "fragment_weights" as an argument
        kwargs.pop(self.fragment_weights_key)

        should_return_tokens = False
        if self.return_tokens_key in kwargs:
            should_return_tokens = kwargs.get(self.return_tokens_key, False)
            # self.transformer doesn't like having extra kwargs
            kwargs.pop(self.return_tokens_key)

        batch_z = None
        batch_tokens = None
        for fragments, weights in zip(text, fragment_weights):

            # First, weight tokens in individual fragments by scaling the feature vectors as requested (effectively
            # applying a multiplier to the CFG scale on a per-token basis).
            # For tokens weighted<1, intuitively we want SD to become not merely *less* interested in the concept
            # captured by the fragment but actually *dis*interested in it (a 0.01 interest in "red" is still an active
            # interest, however small, in redness; what the user probably intends when they attach the number 0.01 to
            # "red" is to tell SD that it should almost completely *ignore* redness).
            # To do this, the embedding is lerped away from base_embedding in the direction of an embedding for a prompt
            # string from which the low-weighted fragment has been simply removed. The closer the weight is to zero, the
            # closer the resulting embedding is to an embedding for a prompt that simply lacks this fragment.

            # handle weights >=1
            tokens, per_token_weights = self.get_tokens_and_weights(fragments, weights)
            base_embedding = self.build_weighted_embedding_tensor(tokens, per_token_weights, **kwargs)

            # this is our starting point
            embeddings = base_embedding.unsqueeze(0)
            per_embedding_weights = [1.0]

            # now handle weights <1
            # Do this by building extra embeddings tensors that lack the words being <1 weighted. These will be lerped
            # with the embeddings tensors that have the words, such that if the weight of a word is 0.5, the resulting
            # embedding will be exactly half-way between the unweighted prompt and the prompt with the <1 weighted words
            # removed.
            # eg for "mountain:1 man:0.5", intuitively the "man" should be "half-gone". therefore, append an embedding
            # for "mountain" (i.e. without "man") to the already-produced embedding for "mountain man", and weight it
            # such that the resulting lerped embedding is exactly half-way between "mountain man" and "mountain".
            for index, fragment_weight in enumerate(weights):
                if fragment_weight < 1:
                    fragments_without_this = fragments[:index] + fragments[index+1:]
                    weights_without_this = weights[:index] + weights[index+1:]
                    tokens, per_token_weights = self.get_tokens_and_weights(fragments_without_this, weights_without_this)
                    embedding_without_this = self.build_weighted_embedding_tensor(tokens, per_token_weights, **kwargs)

                    embeddings = torch.cat((embeddings, embedding_without_this.unsqueeze(0)), dim=1)
                    # weight of the embedding *without* this fragment gets *stronger* as its weight approaches 0
                    # if fragment_weight = 0, basically we want embedding_without_this to completely overwhelm base_embedding
                    # therefore:
                    # fragment_weight = 1: we are at base_z => lerp weight 0
                    # fragment_weight = 0.5: we are halfway between base_z and here => lerp weight 1
                    # fragment_weight = 0: we're now entirely overriding base_z ==> lerp weight inf
                    # so let's use tan(), because:
                    # tan is 0.0 at 0,
                    #        1.0 at PI/4, and
                    #        inf at PI/2
                    # -> tan((1-weight)*PI/2) should give us ideal lerp weights
                    epsilon = 1e-9
                    fragment_weight = max(epsilon, fragment_weight) # inf is bad
                    embedding_lerp_weight = math.tan((1.0 - fragment_weight) * math.pi / 2)
                    # todo handle negative weight?

                    per_embedding_weights.append(embedding_lerp_weight)

            lerped_embeddings = self.apply_embedding_weights(embeddings, per_embedding_weights, normalize=True).squeeze(0)

            #print(f"assembled tokens for '{fragments}' into tensor of shape {lerped_embeddings.shape}")

            # append to batch
            batch_z = lerped_embeddings.unsqueeze(0) if batch_z is None else torch.cat([batch_z, lerped_embeddings.unsqueeze(0)], dim=1)
            batch_tokens = tokens.unsqueeze(0) if batch_tokens is None else torch.cat([batch_tokens, tokens.unsqueeze(0)], dim=1)

        # should have shape (B, 77, 768)
        #print(f"assembled all tokens into tensor of shape {batch_z.shape}")

        if should_return_tokens:
            return batch_z, batch_tokens
        else:
            return batch_z

    def get_token_ids(self, fragments: list[str], include_start_and_end_markers: bool = True) -> list[list[int]]:
        """
        Convert a list of strings like `["a cat", "sitting", "on a mat"]` into a list of lists of token ids like
        `[[bos, 0, 1, eos], [bos, 2, eos], [bos, 3, 0, 4, eos]]`. bos/eos markers are skipped if
        `include_start_and_end_markers` is `False`. Each list will be restricted to the maximum permitted length
        (typically 75 tokens + eos/bos markers).

        :param fragments: The strings to convert.
        :param include_start_and_end_markers:
        :return:
        """

        # for args documentation see ENCODE_KWARGS_DOCSTRING in tokenization_utils_base.py (in `transformers` lib)
        token_ids_list = self.tokenizer(
            fragments,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding='do_not_pad',
            return_tensors=None,  # just give me lists of ints
        )['input_ids']

        result = []
        for token_ids in token_ids_list:
            # trim eos/bos
            token_ids = token_ids[1:-1]
            # pad for textual inversions with vector length >1
            token_ids = self.textual_inversion_manager.expand_textual_inversion_token_ids_if_necessary(token_ids)
            # restrict length to max_length-2 (leaving room for bos/eos)
            token_ids = token_ids[0:self.max_length - 2]
            # add back eos/bos if requested
            if include_start_and_end_markers:
                token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]

            result.append(token_ids)

        return result


    @classmethod
    def apply_embedding_weights(self, embeddings: torch.Tensor, per_embedding_weights: list[float], normalize:bool) -> torch.Tensor:
        per_embedding_weights = torch.tensor(per_embedding_weights, dtype=embeddings.dtype, device=embeddings.device)
        if normalize:
            per_embedding_weights = per_embedding_weights / torch.sum(per_embedding_weights)
        reshaped_weights = per_embedding_weights.reshape(per_embedding_weights.shape + (1, 1,))
        #reshaped_weights = per_embedding_weights.reshape(per_embedding_weights.shape + (1,1,)).expand(embeddings.shape)
        return torch.sum(embeddings * reshaped_weights, dim=1)
        # lerped embeddings has shape (77, 768)


    def get_tokens_and_weights(self, fragments: list[str], weights: list[float]) -> (torch.Tensor, torch.Tensor):
        '''

        :param fragments:
        :param weights: Per-fragment weights (CFG scaling). No need for these to be normalized. They will not be normalized here and that's fine.
        :return:
        '''
        # empty is meaningful
        if len(fragments) == 0 and len(weights) == 0:
            fragments = ['']
            weights = [1]
        per_fragment_token_ids = self.get_token_ids(fragments, include_start_and_end_markers=False)
        all_token_ids = []
        per_token_weights = []
        #print("all fragments:", fragments, weights)
        for index, fragment in enumerate(per_fragment_token_ids):
            weight = float(weights[index])
            #print("processing fragment", fragment, weight)
            this_fragment_token_ids = per_fragment_token_ids[index]
            #print("fragment", fragment, "processed to", this_fragment_token_ids)
            # append
            all_token_ids += this_fragment_token_ids
            # fill out weights tensor with one float per token
            per_token_weights += [weight] * len(this_fragment_token_ids)

        # leave room for bos/eos
        if len(all_token_ids) > self.max_length - 2:
            excess_token_count = len(all_token_ids) - self.max_length - 2
            # TODO build nice description string of how the truncation was applied
            # this should be done by calling self.tokenizer.convert_ids_to_tokens() then passing the result to
            # self.tokenizer.convert_tokens_to_string() for the token_ids on each side of the truncation limit.
            print(f">> Prompt is {excess_token_count} token(s) too long and has been truncated")
            all_token_ids = all_token_ids[0:self.max_length]
            per_token_weights = per_token_weights[0:self.max_length]

        # pad out to a 77-entry array: [eos_token, <prompt tokens>, eos_token, ..., eos_token]
        # (77 = self.max_length)
        all_token_ids = [self.tokenizer.bos_token_id] + all_token_ids + [self.tokenizer.eos_token_id]
        per_token_weights = [1.0] + per_token_weights + [1.0]
        pad_length = self.max_length - len(all_token_ids)
        all_token_ids += [self.tokenizer.eos_token_id] * pad_length
        per_token_weights += [1.0] * pad_length

        all_token_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long).to(self.device)
        per_token_weights_tensor = torch.tensor(per_token_weights, dtype=torch.float32).to(self.device)
        #print(f"assembled all_token_ids_tensor with shape {all_token_ids_tensor.shape}")
        return all_token_ids_tensor, per_token_weights_tensor

    def build_weighted_embedding_tensor(self, token_ids: torch.Tensor, per_token_weights: torch.Tensor, weight_delta_from_empty=True, **kwargs) -> torch.Tensor:
        '''
        Build a tensor representing the passed-in tokens, each of which has a weight.
        :param token_ids: A tensor of shape (77) containing token ids (integers)
        :param per_token_weights: A tensor of shape (77) containing weights (floats)
        :param method: Whether to multiply the whole feature vector for each token or just its distance from an "empty" feature vector
        :param kwargs: passed on to self.transformer()
        :return: A tensor of shape (1, 77, 768) representing the requested weighted embeddings.
        '''
        #print(f"building weighted embedding tensor for {tokens} with weights {per_token_weights}")
        if token_ids.shape != torch.Size([self.max_length]):
            raise ValueError(f"token_ids has shape {token_ids.shape} - expected [{self.max_length}]")

        z = self.transformer(input_ids=token_ids.unsqueeze(0), **kwargs)

        batch_weights_expanded = per_token_weights.reshape(per_token_weights.shape + (1,)).expand(z.shape)

        if weight_delta_from_empty:
            empty_tokens = self.tokenizer([''] * z.shape[0],
                                         truncation=True,
                                         max_length=self.max_length,
                                         padding='max_length',
                                         return_tensors='pt'
                                         )['input_ids'].to(self.device)
            empty_z = self.transformer(input_ids=empty_tokens, **kwargs)
            z_delta_from_empty = z - empty_z
            weighted_z = empty_z + (z_delta_from_empty * batch_weights_expanded)

            #weighted_z_delta_from_empty = (weighted_z-empty_z)
            #print("weighted z has delta from empty with sum", weighted_z_delta_from_empty.sum().item(), "mean", weighted_z_delta_from_empty.mean().item() )

            #print("using empty-delta method, first 5 rows:")
            #print(weighted_z[:5])

            return weighted_z

        else:
            original_mean = z.mean()
            z *= batch_weights_expanded
            after_weighting_mean = z.mean()
            # correct the mean. not sure if this is right but it's what the automatic1111 fork of SD does
            mean_correction_factor = original_mean/after_weighting_mean
            z *= mean_correction_factor
            return z


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(
        self,
        version='ViT-L/14',
        device=choose_torch_device(),
        max_length=77,
        n_repeat=1,
        normalize=True,
    ):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device=device)
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
    Uses the CLIP image encoder.
    """

    def __init__(
        self,
        model,
        jit=False,
        device=choose_torch_device(),
        antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            persistent=False,
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
            persistent=False,
        )

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation='bicubic',
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == '__main__':
    from ldm.util import count_params

    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
