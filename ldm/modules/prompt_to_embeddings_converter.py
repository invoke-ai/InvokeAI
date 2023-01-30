import math

import torch
from transformers import CLIPTokenizer, CLIPTextModel

from ldm.invoke.devices import torch_dtype
from ldm.modules.textual_inversion_manager import TextualInversionManager


class WeightedPromptFragmentsToEmbeddingsConverter():

    def __init__(self,
                tokenizer: CLIPTokenizer, # converts strings to lists of int token ids
                text_encoder: CLIPTextModel, # convert a list of int token ids to a tensor of embeddings
                textual_inversion_manager: TextualInversionManager = None
                ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.textual_inversion_manager = textual_inversion_manager

    @property
    def max_length(self):
        return self.tokenizer.model_max_length

    def get_embeddings_for_weighted_prompt_fragments(self,
                                                     text: list[list[str]],
                                                     fragment_weights: list[list[float]],
                                                     should_return_tokens: bool = False,
                                                     device='cpu'
                                                     ) -> torch.Tensor:
        '''

        :param text: A list of fragments of text to which different weights are to be applied.
        :param fragment_weights: A batch of lists of weights, one for each entry in `fragments`.
        :return: A tensor of shape `[1, 77, token_dim]` containing weighted embeddings where token_dim is 768 for SD1
                    and 1280 for SD2
        '''
        if len(text) != len(fragment_weights):
            raise ValueError(f"lengths of text and fragment_weights lists are not the same ({len(text)} != {len(fragment_weights)})")

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
            tokens, per_token_weights = self.get_token_ids_and_expand_weights(fragments, weights, device=device)
            base_embedding = self.build_weighted_embedding_tensor(tokens, per_token_weights)

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
                    tokens, per_token_weights = self.get_token_ids_and_expand_weights(fragments_without_this, weights_without_this, device=device)
                    embedding_without_this = self.build_weighted_embedding_tensor(tokens, per_token_weights)

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


    def get_token_ids_and_expand_weights(self, fragments: list[str], weights: list[float], device: str) -> (torch.Tensor, torch.Tensor):
        '''
        Given a list of text fragments and corresponding weights: tokenize each fragment, append the token sequences
        together and return a padded token sequence starting with the bos marker, ending with the eos marker, and padded
        or truncated as appropriate to `self.max_length`. Also return a list of weights expanded from the passed-in
        weights to match each token.

        :param fragments: Text fragments to tokenize and concatenate. May be empty.
        :param weights: Per-fragment weights (i.e. quasi-CFG scaling). Values from 0 to inf are permitted. In practise with SD1.5
                        values >1.6 tend to produce garbage output. Must have same length as `fragment`.
        :return: A tuple of tensors `(token_ids, weights)`. `token_ids` is ints, `weights` is floats, both have shape `[self.max_length]`.
        '''
        if len(fragments) != len(weights):
            raise ValueError(f"lengths of text and fragment_weights lists are not the same ({len(fragments)} != {len(weights)})")

        # empty is meaningful
        if len(fragments) == 0:
            fragments = ['']
            weights = [1.0]
        per_fragment_token_ids = self.get_token_ids(fragments, include_start_and_end_markers=False)
        all_token_ids = []
        per_token_weights = []
        #print("all fragments:", fragments, weights)
        for this_fragment_token_ids, weight in zip(per_fragment_token_ids, weights):
            # append
            all_token_ids += this_fragment_token_ids
            # fill out weights tensor with one float per token
            per_token_weights += [float(weight)] * len(this_fragment_token_ids)

        # leave room for bos/eos
        max_token_count_without_bos_eos_markers = self.max_length - 2
        if len(all_token_ids) > max_token_count_without_bos_eos_markers:
            excess_token_count = len(all_token_ids) - max_token_count_without_bos_eos_markers
            # TODO build nice description string of how the truncation was applied
            # this should be done by calling self.tokenizer.convert_ids_to_tokens() then passing the result to
            # self.tokenizer.convert_tokens_to_string() for the token_ids on each side of the truncation limit.
            print(f">> Prompt is {excess_token_count} token(s) too long and has been truncated")
            all_token_ids = all_token_ids[0:max_token_count_without_bos_eos_markers]
            per_token_weights = per_token_weights[0:max_token_count_without_bos_eos_markers]

        # pad out to a self.max_length-entry array: [bos_token, <prompt tokens>, eos_token, pad_token…]
        # (typically self.max_length == 77)
        all_token_ids = [self.tokenizer.bos_token_id] + all_token_ids + [self.tokenizer.eos_token_id]
        per_token_weights = [1.0] + per_token_weights + [1.0]
        pad_length = self.max_length - len(all_token_ids)
        all_token_ids += [self.tokenizer.pad_token_id] * pad_length
        per_token_weights += [1.0] * pad_length

        all_token_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long, device=device)
        per_token_weights_tensor = torch.tensor(per_token_weights, dtype=torch_dtype(self.text_encoder.device), device=device)
        #print(f"assembled all_token_ids_tensor with shape {all_token_ids_tensor.shape}")
        return all_token_ids_tensor, per_token_weights_tensor

    def build_weighted_embedding_tensor(self, token_ids: torch.Tensor, per_token_weights: torch.Tensor) -> torch.Tensor:
        '''
        Build a tensor that embeds the passed-in token IDs and applyies the given per_token weights
        :param token_ids: A tensor of shape `[self.max_length]` containing token IDs (ints)
        :param per_token_weights: A tensor of shape `[self.max_length]` containing weights (floats)
        :return: A tensor of shape `[1, self.max_length, token_dim]` representing the requested weighted embeddings
        where `token_dim` is 768 for SD1 and 1280 for SD2.
        '''
        #print(f"building weighted embedding tensor for {tokens} with weights {per_token_weights}")
        if token_ids.shape != torch.Size([self.max_length]):
            raise ValueError(f"token_ids has shape {token_ids.shape} - expected [{self.max_length}]")

        z = self.text_encoder.forward(input_ids=token_ids.unsqueeze(0),
                                      return_dict=False)[0]
        empty_token_ids = torch.tensor([self.tokenizer.bos_token_id] +
                                    [self.tokenizer.pad_token_id] * (self.max_length-2) +
                                    [self.tokenizer.eos_token_id], dtype=torch.int, device=token_ids.device).unsqueeze(0)
        empty_z = self.text_encoder(input_ids=empty_token_ids).last_hidden_state
        batch_weights_expanded = per_token_weights.reshape(per_token_weights.shape + (1,)).expand(z.shape)
        z_delta_from_empty = z - empty_z
        weighted_z = empty_z + (z_delta_from_empty * batch_weights_expanded)

        return weighted_z
