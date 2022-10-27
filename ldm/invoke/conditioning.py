'''
This module handles the generation of the conditioning tensors, including management of
weighted subprompts.

Useful function exports:

get_uc_and_c_and_ec()           get the conditioned and unconditioned latent, and edited conditioning if we're doing cross-attention control
split_weighted_subpromopts()    split subprompts, normalize and weight them
log_tokenization()              print out colour-coded tokens and warn if truncated

'''
import re
from difflib import SequenceMatcher
from typing import Union

import torch

from .prompt_parser import PromptParser, Blend, FlattenedPrompt, \
    CrossAttentionControlledFragment, CrossAttentionControlSubstitute, CrossAttentionControlAppend, Fragment
from ..models.diffusion.cross_attention_control import CrossAttentionControl
from ..models.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
from ..modules.encoders.modules import WeightedFrozenCLIPEmbedder


def get_uc_and_c_and_ec(prompt_string_uncleaned, model, log_tokens=False, skip_normalize=False):

    # Extract Unconditioned Words From Prompt
    unconditioned_words = ''
    unconditional_regex = r'\[(.*?)\]'
    unconditionals = re.findall(unconditional_regex, prompt_string_uncleaned)

    if len(unconditionals) > 0:
        unconditioned_words = ' '.join(unconditionals)

        # Remove Unconditioned Words From Prompt
        unconditional_regex_compile = re.compile(unconditional_regex)
        clean_prompt = unconditional_regex_compile.sub(' ', prompt_string_uncleaned)
        prompt_string_cleaned = re.sub(' +', ' ', clean_prompt)
    else:
        prompt_string_cleaned = prompt_string_uncleaned

    pp = PromptParser()

    parsed_prompt: Union[FlattenedPrompt, Blend] = None
    legacy_blend: Blend = pp.parse_legacy_blend(prompt_string_cleaned)
    if legacy_blend is not None:
        parsed_prompt = legacy_blend
    else:
        # we don't support conjunctions for now
        parsed_prompt = pp.parse_conjunction(prompt_string_cleaned).prompts[0]

    parsed_negative_prompt: FlattenedPrompt = pp.parse_conjunction(unconditioned_words).prompts[0]
    print(f">> Parsed prompt to {parsed_prompt}")

    conditioning = None
    cac_args:CrossAttentionControl.Arguments = None

    if type(parsed_prompt) is Blend:
        blend: Blend = parsed_prompt
        embeddings_to_blend = None
        for flattened_prompt in blend.prompts:
            this_embedding, _ = build_embeddings_and_tokens_for_flattened_prompt(model, flattened_prompt, log_tokens=log_tokens)
            embeddings_to_blend = this_embedding if embeddings_to_blend is None else torch.cat(
                (embeddings_to_blend, this_embedding))
        conditioning = WeightedFrozenCLIPEmbedder.apply_embedding_weights(embeddings_to_blend.unsqueeze(0),
                                                                                blend.weights,
                                                                                normalize=blend.normalize_weights)
    else:
        flattened_prompt: FlattenedPrompt = parsed_prompt
        wants_cross_attention_control = type(flattened_prompt) is not Blend \
                                        and any([issubclass(type(x), CrossAttentionControlledFragment) for x in flattened_prompt.children])
        if wants_cross_attention_control:
            original_prompt = FlattenedPrompt()
            edited_prompt = FlattenedPrompt()
            # for name, a0, a1, b0, b1 in edit_opcodes: only name == 'equal' is currently parsed
            original_token_count = 0
            edited_token_count = 0
            edit_opcodes = []
            edit_options = []
            for fragment in flattened_prompt.children:
                if type(fragment) is CrossAttentionControlSubstitute:
                    original_prompt.append(fragment.original)
                    edited_prompt.append(fragment.edited)

                    to_replace_token_count = get_tokens_length(model, fragment.original)
                    replacement_token_count = get_tokens_length(model, fragment.edited)
                    edit_opcodes.append(('replace',
                                        original_token_count, original_token_count + to_replace_token_count,
                                        edited_token_count, edited_token_count + replacement_token_count
                                        ))
                    original_token_count += to_replace_token_count
                    edited_token_count += replacement_token_count
                    edit_options.append(fragment.options)
                #elif type(fragment) is CrossAttentionControlAppend:
                #    edited_prompt.append(fragment.fragment)
                else:
                    # regular fragment
                    original_prompt.append(fragment)
                    edited_prompt.append(fragment)

                    count = get_tokens_length(model, [fragment])
                    edit_opcodes.append(('equal', original_token_count, original_token_count+count, edited_token_count, edited_token_count+count))
                    edit_options.append(None)
                    original_token_count += count
                    edited_token_count += count
            original_embeddings, original_tokens = build_embeddings_and_tokens_for_flattened_prompt(model, original_prompt, log_tokens=log_tokens)
            # naïvely building a single edited_embeddings like this disregards the effects of changing the absolute location of
            # subsequent tokens when there is >1 edit and earlier edits change the total token count.
            # eg "a cat.swap(smiling dog, s_start=0.5) eating a hotdog.swap(pizza)" - when the 'pizza' edit is active but the
            # 'cat' edit is not, the 'pizza' feature vector will nevertheless be affected by the introduction of the extra
            # token 'smiling' in the inactive 'cat' edit.
            # todo: build multiple edited_embeddings, one for each edit, and pass just the edited fragments through to the CrossAttentionControl functions
            edited_embeddings, edited_tokens = build_embeddings_and_tokens_for_flattened_prompt(model, edited_prompt, log_tokens=log_tokens)

            conditioning = original_embeddings
            edited_conditioning = edited_embeddings
            print('got edit_opcodes', edit_opcodes, 'options', edit_options)
            cac_args = CrossAttentionControl.Arguments(
                edited_conditioning = edited_conditioning,
                edit_opcodes = edit_opcodes,
                edit_options = edit_options
            )
        else:
            conditioning, _ = build_embeddings_and_tokens_for_flattened_prompt(model, flattened_prompt, log_tokens=log_tokens)

    unconditioning, _ = build_embeddings_and_tokens_for_flattened_prompt(model, parsed_negative_prompt, log_tokens=log_tokens)
    conditioning = flatten_hybrid_conditioning(unconditioning, conditioning)
    return (
        unconditioning, conditioning, InvokeAIDiffuserComponent.ExtraConditioningInfo(
            cross_attention_control_args=cac_args
        )
    )


def build_token_edit_opcodes(original_tokens, edited_tokens):
    original_tokens = original_tokens.cpu().numpy()[0]
    edited_tokens = edited_tokens.cpu().numpy()[0]

    return SequenceMatcher(None, original_tokens, edited_tokens).get_opcodes()

def build_embeddings_and_tokens_for_flattened_prompt(model, flattened_prompt: FlattenedPrompt, log_tokens: bool=False):
    if type(flattened_prompt) is not FlattenedPrompt:
        raise Exception(f"embeddings can only be made from FlattenedPrompts, got {type(flattened_prompt)} instead")
    fragments = [x.text for x in flattened_prompt.children]
    weights = [x.weight for x in flattened_prompt.children]
    embeddings, tokens = model.get_learned_conditioning([fragments], return_tokens=True, fragment_weights=[weights])
    if not flattened_prompt.is_empty and log_tokens:
        start_token = model.cond_stage_model.tokenizer.bos_token_id
        end_token = model.cond_stage_model.tokenizer.eos_token_id
        tokens_list = tokens[0].tolist()
        if tokens_list[0] == start_token:
            tokens_list[0] = '<start>'
        try:
            first_end_token_index = tokens_list.index(end_token)
            tokens_list[first_end_token_index] = '<end>'
            tokens_list = tokens_list[:first_end_token_index+1]
        except ValueError:
            pass

        print(f">> Prompt fragments {fragments}, tokenized to \n{tokens_list}")

    return embeddings, tokens

def get_tokens_length(model, fragments: list[Fragment]):
    fragment_texts = [x.text for x in fragments]
    tokens = model.cond_stage_model.get_tokens(fragment_texts, include_start_and_end_markers=False)
    return sum([len(x) for x in tokens])

def flatten_hybrid_conditioning(uncond, cond):
    '''
    This handles the choice between a conditional conditioning
    that is a tensor (used by cross attention) vs one that has additional
    dimensions as well, as used by 'hybrid'
    '''
    if isinstance(cond, dict):
        assert isinstance(uncond, dict)
        cond_in = dict()
        for k in cond:
            if isinstance(cond[k], list):
                cond_in[k] = [
                    torch.cat([uncond[k][i], cond[k][i]])
                    for i in range(len(cond[k]))
                ]
            else:
                cond_in[k] = torch.cat([uncond[k], cond[k]])
        return cond_in
    else:
        return cond

            
