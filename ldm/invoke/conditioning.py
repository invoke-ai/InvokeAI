'''
This module handles the generation of the conditioning tensors.

Useful function exports:

get_uc_and_c_and_ec()           get the conditioned and unconditioned latent, and edited conditioning if we're doing cross-attention control

'''
import re
from typing import Union

import torch

from .prompt_parser import PromptParser, Blend, FlattenedPrompt, \
    CrossAttentionControlledFragment, CrossAttentionControlSubstitute, Fragment
from ..models.diffusion import cross_attention_control
from ..models.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
from ..modules.encoders.modules import WeightedFrozenCLIPEmbedder
from ..modules.prompt_to_embeddings_converter import WeightedPromptFragmentsToEmbeddingsConverter
from ldm.invoke.globals import Globals


def get_uc_and_c_and_ec(prompt_string, model, log_tokens=False, skip_normalize_legacy_blend=False):

    # lazy-load any deferred textual inversions.
    # this might take a couple of seconds the first time a textual inversion is used.
    model.textual_inversion_manager.create_deferred_token_ids_for_any_trigger_terms(prompt_string)

    prompt, negative_prompt = get_prompt_structure(prompt_string,
                                                   skip_normalize_legacy_blend=skip_normalize_legacy_blend)
    conditioning = _get_conditioning_for_prompt(prompt, negative_prompt, model, log_tokens)

    return conditioning


def get_prompt_structure(prompt_string, skip_normalize_legacy_blend: bool = False) -> (
Union[FlattenedPrompt, Blend], FlattenedPrompt):
    """
    parse the passed-in prompt string and return tuple (positive_prompt, negative_prompt)
    """
    prompt, negative_prompt = _parse_prompt_string(prompt_string,
                                                   skip_normalize_legacy_blend=skip_normalize_legacy_blend)
    return prompt, negative_prompt


def get_tokens_for_prompt(model, parsed_prompt: FlattenedPrompt, truncate_if_too_long=True) -> [str]:
    text_fragments = [x.text if type(x) is Fragment else
                      (" ".join([f.text for f in x.original]) if type(x) is CrossAttentionControlSubstitute else
                       str(x))
                      for x in parsed_prompt.children]
    text = " ".join(text_fragments)
    tokens = model.cond_stage_model.tokenizer.tokenize(text)
    if truncate_if_too_long:
        max_tokens_length = model.cond_stage_model.max_length - 2 # typically 75
        tokens = tokens[0:max_tokens_length]
    return tokens


def _parse_prompt_string(prompt_string_uncleaned, skip_normalize_legacy_blend=False) -> Union[FlattenedPrompt, Blend]:
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
    legacy_blend: Blend = pp.parse_legacy_blend(prompt_string_cleaned, skip_normalize_legacy_blend)
    if legacy_blend is not None:
        parsed_prompt = legacy_blend
    else:
        # we don't support conjunctions for now
        parsed_prompt = pp.parse_conjunction(prompt_string_cleaned).prompts[0]

    parsed_negative_prompt: FlattenedPrompt = pp.parse_conjunction(unconditioned_words).prompts[0]
    return parsed_prompt, parsed_negative_prompt


def _get_conditioning_for_prompt(parsed_prompt: Union[Blend, FlattenedPrompt], parsed_negative_prompt: FlattenedPrompt,
                                 model, log_tokens=False) \
    -> tuple[torch.Tensor, torch.Tensor, InvokeAIDiffuserComponent.ExtraConditioningInfo]:
    """
    Process prompt structure and tokens, and return (conditioning, unconditioning, extra_conditioning_info)
    """

    if log_tokens or Globals.log_tokenization:
        print(f"\n>> [TOKENLOG] Parsed Prompt: {parsed_prompt}")
        print(f"\n>> [TOKENLOG] Parsed Negative Prompt: {parsed_negative_prompt}")

    conditioning = None
    cac_args: cross_attention_control.Arguments = None

    if type(parsed_prompt) is Blend:
        conditioning = _get_conditioning_for_blend(model, parsed_prompt, log_tokens)
    elif type(parsed_prompt) is FlattenedPrompt:
        if parsed_prompt.wants_cross_attention_control:
            conditioning, cac_args = _get_conditioning_for_cross_attention_control(model, parsed_prompt, log_tokens)

        else:
            conditioning, _ = _get_embeddings_and_tokens_for_prompt(model,
                                                                    parsed_prompt,
                                                                    log_tokens=log_tokens,
                                                                    log_display_label="(prompt)")
    else:
        raise ValueError(f"parsed_prompt is '{type(parsed_prompt)}' which is not a supported prompt type")

    unconditioning, _ = _get_embeddings_and_tokens_for_prompt(model,
                                                              parsed_negative_prompt,
                                                              log_tokens=log_tokens,
                                                              log_display_label="(unconditioning)")
    if isinstance(conditioning, dict):
        # hybrid conditioning is in play
        unconditioning, conditioning = _flatten_hybrid_conditioning(unconditioning, conditioning)
        if cac_args is not None:
            print(
                ">> Hybrid conditioning cannot currently be combined with cross attention control. Cross attention control will be ignored.")
            cac_args = None

    if type(parsed_prompt) is Blend:
        blend: Blend = parsed_prompt
        all_token_sequences = [get_tokens_for_prompt(model, p) for p in blend.prompts]
        longest_token_sequence = max(all_token_sequences, key=lambda t: len(t))
        eos_token_index = len(longest_token_sequence)+1
    else:
        tokens = get_tokens_for_prompt(model, parsed_prompt)
        eos_token_index = len(tokens)+1
    return (
        unconditioning, conditioning, InvokeAIDiffuserComponent.ExtraConditioningInfo(
            tokens_count_including_eos_bos=eos_token_index + 1,
            cross_attention_control_args=cac_args
        )
    )


def _get_conditioning_for_cross_attention_control(model, prompt: FlattenedPrompt, log_tokens: bool = True):
    original_prompt = FlattenedPrompt()
    edited_prompt = FlattenedPrompt()
    # for name, a0, a1, b0, b1 in edit_opcodes: only name == 'equal' is currently parsed
    original_token_count = 0
    edited_token_count = 0
    edit_options = []
    edit_opcodes = []
    # beginning of sequence
    edit_opcodes.append(
        ('equal', original_token_count, original_token_count + 1, edited_token_count, edited_token_count + 1))
    edit_options.append(None)
    original_token_count += 1
    edited_token_count += 1
    for fragment in prompt.children:
        if type(fragment) is CrossAttentionControlSubstitute:
            original_prompt.append(fragment.original)
            edited_prompt.append(fragment.edited)

            to_replace_token_count = _get_tokens_length(model, fragment.original)
            replacement_token_count = _get_tokens_length(model, fragment.edited)
            edit_opcodes.append(('replace',
                                 original_token_count, original_token_count + to_replace_token_count,
                                 edited_token_count, edited_token_count + replacement_token_count
                                 ))
            original_token_count += to_replace_token_count
            edited_token_count += replacement_token_count
            edit_options.append(fragment.options)
        # elif type(fragment) is CrossAttentionControlAppend:
        #    edited_prompt.append(fragment.fragment)
        else:
            # regular fragment
            original_prompt.append(fragment)
            edited_prompt.append(fragment)

            count = _get_tokens_length(model, [fragment])
            edit_opcodes.append(('equal', original_token_count, original_token_count + count, edited_token_count,
                                 edited_token_count + count))
            edit_options.append(None)
            original_token_count += count
            edited_token_count += count
    # end of sequence
    edit_opcodes.append(
        ('equal', original_token_count, original_token_count + 1, edited_token_count, edited_token_count + 1))
    edit_options.append(None)
    original_token_count += 1
    edited_token_count += 1
    original_embeddings, original_tokens = _get_embeddings_and_tokens_for_prompt(model,
                                                                                 original_prompt,
                                                                                 log_tokens=log_tokens,
                                                                                 log_display_label="(.swap originals)")
    # naïvely building a single edited_embeddings like this disregards the effects of changing the absolute location of
    # subsequent tokens when there is >1 edit and earlier edits change the total token count.
    # eg "a cat.swap(smiling dog, s_start=0.5) eating a hotdog.swap(pizza)" - when the 'pizza' edit is active but the
    # 'cat' edit is not, the 'pizza' feature vector will nevertheless be affected by the introduction of the extra
    # token 'smiling' in the inactive 'cat' edit.
    # todo: build multiple edited_embeddings, one for each edit, and pass just the edited fragments through to the CrossAttentionControl functions
    edited_embeddings, edited_tokens = _get_embeddings_and_tokens_for_prompt(model,
                                                                             edited_prompt,
                                                                             log_tokens=log_tokens,
                                                                             log_display_label="(.swap replacements)")
    conditioning = original_embeddings
    edited_conditioning = edited_embeddings
    # print('>> got edit_opcodes', edit_opcodes, 'options', edit_options)
    cac_args = cross_attention_control.Arguments(
        edited_conditioning=edited_conditioning,
        edit_opcodes=edit_opcodes,
        edit_options=edit_options
    )
    return conditioning, cac_args


def _get_conditioning_for_blend(model, blend: Blend, log_tokens: bool = False):
    embeddings_to_blend = None
    for i, flattened_prompt in enumerate(blend.prompts):
        this_embedding, _ = _get_embeddings_and_tokens_for_prompt(model,
                                                                  flattened_prompt,
                                                                  log_tokens=log_tokens,
                                                                  log_display_label=f"(blend part {i + 1}, weight={blend.weights[i]})")
        embeddings_to_blend = this_embedding if embeddings_to_blend is None else torch.cat(
            (embeddings_to_blend, this_embedding))
    conditioning = WeightedPromptFragmentsToEmbeddingsConverter.apply_embedding_weights(embeddings_to_blend.unsqueeze(0),
                                                                      blend.weights,
                                                                      normalize=blend.normalize_weights)
    return conditioning


def _get_embeddings_and_tokens_for_prompt(model, flattened_prompt: FlattenedPrompt, log_tokens: bool = False,
                                          log_display_label: str = None):
    if type(flattened_prompt) is not FlattenedPrompt:
        raise Exception(f"embeddings can only be made from FlattenedPrompts, got {type(flattened_prompt)} instead")
    fragments = [x.text for x in flattened_prompt.children]
    weights = [x.weight for x in flattened_prompt.children]
    embeddings, tokens = model.get_learned_conditioning([fragments], return_tokens=True, fragment_weights=[weights])
    if log_tokens or Globals.log_tokenization:
        text = " ".join(fragments)
        log_tokenization(text, model, display_label=log_display_label)

    return embeddings, tokens


def _get_tokens_length(model, fragments: list[Fragment]):
    fragment_texts = [x.text for x in fragments]
    tokens = model.cond_stage_model.get_token_ids(fragment_texts, include_start_and_end_markers=False)
    return sum([len(x) for x in tokens])


def _flatten_hybrid_conditioning(uncond, cond):
    '''
    This handles the choice between a conditional conditioning
    that is a tensor (used by cross attention) vs one that has additional
    dimensions as well, as used by 'hybrid'
    '''
    assert isinstance(uncond, dict)
    assert isinstance(cond, dict)
    cond_flattened = dict()
    for k in cond:
        if isinstance(cond[k], list):
            cond_flattened[k] = [
                torch.cat([uncond[k][i], cond[k][i]])
                for i in range(len(cond[k]))
            ]
        else:
            cond_flattened[k] = torch.cat([uncond[k], cond[k]])
    return uncond, cond_flattened


def log_tokenization(text, model, display_label=None):
    """ shows how the prompt is tokenized
    # usually tokens have '</w>' to indicate end-of-word,
    # but for readability it has been replaced with ' '
    """
    tokens = model.cond_stage_model.tokenizer.tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)

    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"

    if usedTokens > 0:
        print(f'\n>> [TOKENLOG] Tokens {display_label or ""} ({usedTokens}):')
        print(f'{tokenized}\x1b[0m')
    
    if discarded != "":
        print(f'\n>> [TOKENLOG] Tokens Discarded ({totalTokens - usedTokens}):')
        print(f'{discarded}\x1b[0m')