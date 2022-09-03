'''
This module handles the generation of the conditioning tensors, including management of
weighted subprompts.
'''
import re
import torch

class Conditioning():
    def __init__(self, model, logger=None):
        self.model  = model
        self.logger = logger if logger else lambda : None    # right way to make a noop?

    def get_uc_and_c(self, prompt, skip_normalize=False):
        uc = self.model.get_learned_conditioning([''])

        # get weighted sub-prompts
        weighted_subprompts = self.split_weighted_subprompts(
            prompt, skip_normalize
        )

        if len(weighted_subprompts) > 1:
            # i dont know if this is correct.. but it works
            c = torch.zeros_like(uc)
            # normalize each "sub prompt" and add it
            for subprompt, weight in weighted_subprompts:
                self.logger(subprompt)
                c = torch.add(
                    c,
                    self.model.get_learned_conditioning([subprompt]),
                    alpha=weight,
                )
        else:   # just standard 1 prompt
            self.loglogger(prompt)
            c = self.model.get_learned_conditioning([prompt])
        return (uc, c)

    def split_weighted_subprompts(self, text, skip_normalize=False)->list:
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
            equal_weight = 1 / len(parsed_prompts)
            return [(x[0], equal_weight) for x in parsed_prompts]
        return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

        

