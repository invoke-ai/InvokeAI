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

    # we don't support conjunctions for now
    parsed_prompt: Union[FlattenedPrompt, Blend] = pp.parse(prompt_string_cleaned).prompts[0]
    parsed_negative_prompt: FlattenedPrompt = pp.parse(unconditioned_words).prompts[0]
    print("parsed prompt to", parsed_prompt)

    conditioning = None
    edited_conditioning = None
    edit_opcodes = None

    if type(parsed_prompt) is Blend:
        blend: Blend = parsed_prompt
        embeddings_to_blend = None
        for flattened_prompt in blend.prompts:
            this_embedding, _ = build_embeddings_and_tokens_for_flattened_prompt(model, flattened_prompt)
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
            original_embeddings, original_tokens = build_embeddings_and_tokens_for_flattened_prompt(model, original_prompt)
            edited_embeddings, edited_tokens = build_embeddings_and_tokens_for_flattened_prompt(model, edited_prompt)

            conditioning = original_embeddings
            edited_conditioning = edited_embeddings
            print('got edit_opcodes', edit_opcodes, 'options', edit_options)
        else:
            conditioning, _ = build_embeddings_and_tokens_for_flattened_prompt(model, flattened_prompt)


    unconditioning, _ = build_embeddings_and_tokens_for_flattened_prompt(model, parsed_negative_prompt)
    return (
        unconditioning, conditioning, edited_conditioning, edit_opcodes
        #InvokeAIDiffuserComponent.ExtraConditioningInfo(edited_conditioning=edited_conditioning,
        #                                                edit_opcodes=edit_opcodes,
        #                                                edit_options=edit_options)
    )


def build_token_edit_opcodes(original_tokens, edited_tokens):
    original_tokens = original_tokens.cpu().numpy()[0]
    edited_tokens = edited_tokens.cpu().numpy()[0]

    return SequenceMatcher(None, original_tokens, edited_tokens).get_opcodes()

def build_embeddings_and_tokens_for_flattened_prompt(model, flattened_prompt: FlattenedPrompt):
    if type(flattened_prompt) is not FlattenedPrompt:
        raise Exception(f"embeddings can only be made from FlattenedPrompts, got {type(flattened_prompt)} instead")
    fragments = [x.text for x in flattened_prompt.children]
    weights = [x.weight for x in flattened_prompt.children]
    embeddings, tokens = model.get_learned_conditioning([fragments], return_tokens=True, fragment_weights=[weights])
    return embeddings, tokens

def get_tokens_length(model, fragments: list[Fragment]):
    fragment_texts = [x.text for x in fragments]
    tokens = model.cond_stage_model.get_tokens(fragment_texts, include_start_and_end_markers=False)
    return sum([len(x) for x in tokens])


def split_weighted_subprompts(text, skip_normalize=False)->list:
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    prompt_parser = re.compile("""
            (?P<prompt>     # capture group for 'prompt'
            (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
            )               # end 'prompt'
            (?:             # non-capture group
            :+              # match one or more ':' characters
            (?P<weight>     # capture group for 'weight'
            -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
            )?              # end weight capture group, make optional
            \s*             # strip spaces after weight
            |               # OR
            $               # else, if no ':' then match end of line
            )               # end non-capture group
            """, re.VERBOSE)
    parsed_prompts = [(match.group("prompt").replace("\\:", ":"), float(
        match.group("weight") or 1)) for match in re.finditer(prompt_parser, text)]
    if skip_normalize:
        return parsed_prompts
    weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    if weight_sum == 0:
        print(
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, log=False, weight=1):
    if not log:
        return
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
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
    print(f"\n>> Tokens ({usedTokens}), Weight ({weight:.2f}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m"
        )
