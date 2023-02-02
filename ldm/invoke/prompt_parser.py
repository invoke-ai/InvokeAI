import string
from typing import Union, Optional
import re
import pyparsing as pp
'''
This module parses prompt strings and produces tree-like structures that can be used generate and control the conditioning tensors.
weighted subprompts.

Useful class exports:

PromptParser - parses prompts

Useful function exports:

split_weighted_subpromopts()    split subprompts, normalize and weight them
log_tokenization()              print out colour-coded tokens and warn if truncated
'''

class Prompt():
    """
    Mid-level structure for storing the tree-like result of parsing a prompt. A Prompt may not represent the whole of
    the singular user-defined "prompt string" (although it can) - for example, if the user specifies a Blend, the objects
    that are to be blended together are stored individuall as Prompt objects.

    Nesting makes this object not suitable for directly tokenizing; instead call flatten() on the containing Conjunction
    to produce a FlattenedPrompt.
    """
    def __init__(self, parts: list):
        for c in parts:
            if type(c) is not Attention and not issubclass(type(c), BaseFragment) and type(c) is not pp.ParseResults:
                raise PromptParser.ParsingException(f"Prompt cannot contain {type(c).__name__} ({c}), only {[c.__name__ for c in BaseFragment.__subclasses__()]} are allowed")
        self.children = parts
    def __repr__(self):
        return f"Prompt:{self.children}"
    def __eq__(self, other):
        return type(other) is Prompt and other.children == self.children

class BaseFragment:
    pass

class FlattenedPrompt():
    """
    A Prompt that has been passed through flatten(). Its children can be readily tokenized.
    """
    def __init__(self, parts: list=[]):
        self.children = []
        for part in parts:
            self.append(part)

    def append(self, fragment: Union[list, BaseFragment, tuple]):
        # verify type correctness
        if type(fragment) is list:
            for x in fragment:
                self.append(x)
        elif issubclass(type(fragment), BaseFragment):
            self.children.append(fragment)
        elif type(fragment) is tuple:
            # upgrade tuples to Fragments
            if type(fragment[0]) is not str or (type(fragment[1]) is not float and type(fragment[1]) is not int):
                raise PromptParser.ParsingException(
                    f"FlattenedPrompt cannot contain {fragment}, only Fragments or (str, float) tuples are allowed")
            self.children.append(Fragment(fragment[0], fragment[1]))
        else:
            raise PromptParser.ParsingException(
                f"FlattenedPrompt cannot contain {fragment}, only Fragments or (str, float) tuples are allowed")

    @property
    def is_empty(self):
        return len(self.children) == 0 or \
               (len(self.children) == 1 and len(self.children[0].text) == 0)

    @property
    def wants_cross_attention_control(self):
        return any(
            [issubclass(type(x), CrossAttentionControlledFragment) for x in self.children]
        )

    def __repr__(self):
        return f"FlattenedPrompt:{self.children}"
    def __eq__(self, other):
        return type(other) is FlattenedPrompt and other.children == self.children


class Fragment(BaseFragment):
    """
    A Fragment is a chunk of plain text and an optional weight. The text should be passed as-is to the CLIP tokenizer.
    """
    def __init__(self, text: str, weight: float=1):
        assert(type(text) is str)
        if '\\"' in text or '\\(' in text or '\\)' in text:
            #print("Fragment converting escaped \( \) \\\" into ( ) \"")
            text = text.replace('\\(', '(').replace('\\)', ')').replace('\\"', '"')
        self.text = text
        self.weight = float(weight)

    def __repr__(self):
        return "Fragment:'"+self.text+"'@"+str(self.weight)
    def __eq__(self, other):
        return type(other) is Fragment \
            and other.text == self.text \
            and other.weight == self.weight

class Attention():
    """
    Nestable weight control for fragments. Each object in the children array may in turn be an Attention object;
    weights should be considered to accumulate as the tree is traversed to deeper levels of nesting.

    Do not traverse directly; instead obtain a FlattenedPrompt by calling Flatten() on a top-level Conjunction object.
    """
    def __init__(self, weight: float, children: list):
        if type(weight) is not float:
            raise PromptParser.ParsingException(
                f"Attention weight must be float (got {type(weight).__name__} {weight})")
        self.weight = weight
        if type(children) is not list:
            raise PromptParser.ParsingException(f"cannot make Attention with non-list of children (got {type(children)})")
        assert(type(children) is list)
        self.children = children
        #print(f"A: requested attention '{children}' to {weight}")

    def __repr__(self):
        return f"Attention:{self.children} * {self.weight}"
    def __eq__(self, other):
        return type(other) is Attention and other.weight == self.weight and other.fragment == self.fragment

class CrossAttentionControlledFragment(BaseFragment):
    pass

class CrossAttentionControlSubstitute(CrossAttentionControlledFragment):
    """
    A Cross-Attention Controlled ('prompt2prompt') fragment, for use inside a Prompt, Attention, or FlattenedPrompt.
    Representing an "original" word sequence that supplies feature vectors for an initial diffusion operation, and an
    "edited" word sequence, to which the attention maps produced by the "original" word sequence are applied. Intuitively,
    the result should be an "edited" image that looks like the "original" image with concepts swapped.

    eg "a cat sitting on a car" (original) -> "a smiling dog sitting on a car" (edited): the edited image should look
    almost exactly the same as the original, but with a smiling dog rendered in place of the cat. The
    CrossAttentionControlSubstitute object representing this swap may be confined to the tokens being swapped:
        CrossAttentionControlSubstitute(original=[Fragment('cat')], edited=[Fragment('dog')])
    or it may represent a larger portion of the token sequence:
        CrossAttentionControlSubstitute(original=[Fragment('a cat sitting on a car')],
                                        edited=[Fragment('a smiling dog sitting on a car')])

    In either case expect it to be embedded in a Prompt or FlattenedPrompt:
    FlattenedPrompt([
            Fragment('a'),
            CrossAttentionControlSubstitute(original=[Fragment('cat')], edited=[Fragment('dog')]),
            Fragment('sitting on a car')
        ])
    """
    def __init__(self, original: list, edited: list, options: dict=None):
        self.original = original if len(original)>0 else [Fragment('')]
        self.edited = edited if len(edited)>0 else [Fragment('')]

        default_options = {
            's_start': 0.0,
            's_end': 0.2062994740159002, # ~= shape_freedom=0.5
            't_start': 0.1,
            't_end': 1.0
        }
        merged_options = default_options
        if options is not None:
            shape_freedom = options.pop('shape_freedom', None)
            if shape_freedom is not None:
                # high shape freedom = SD can do what it wants with the shape of the object
                # high shape freedom => s_end = 0
                # low shape freedom => s_end = 1
                # shape freedom is in a "linear" space, while noticeable changes to s_end are typically closer around 0,
                # and there is very little perceptible difference as s_end increases above 0.5
                # so for shape_freedom = 0.5 we probably want s_end to be 0.2
                #  -> cube root and subtract from 1.0
                merged_options['s_end'] = 1.0 - shape_freedom ** (1. / 3.)
                #print('converted shape_freedom argument to', merged_options)
            merged_options.update(options)

        self.options = merged_options

    def __repr__(self):
        return f"CrossAttentionControlSubstitute:({self.original}->{self.edited} ({self.options})"
    def __eq__(self, other):
        return type(other) is CrossAttentionControlSubstitute \
               and other.original == self.original \
               and other.edited == self.edited \
               and other.options == self.options


class CrossAttentionControlAppend(CrossAttentionControlledFragment):
    def __init__(self, fragment: Fragment):
        self.fragment = fragment
    def __repr__(self):
        return "CrossAttentionControlAppend:",self.fragment
    def __eq__(self, other):
        return type(other) is CrossAttentionControlAppend \
               and other.fragment == self.fragment



class Conjunction():
    """
    Storage for one or more Prompts or Blends, each of which is to be separately diffused and then the results merged
    by weighted sum in latent space.
    """
    def __init__(self, prompts: list, weights: list = None):
        # force everything to be a Prompt
        #print("making conjunction with", prompts, "types", [type(p).__name__ for p in prompts])
        self.prompts = [x if (type(x) is Prompt
                          or type(x) is Blend
                          or type(x) is FlattenedPrompt)
                      else Prompt(x) for x in prompts]
        self.weights = [1.0]*len(self.prompts) if (weights is None or len(weights)==0) else list(weights)
        if len(self.weights) != len(self.prompts):
            raise PromptParser.ParsingException(f"while parsing Conjunction: mismatched parts/weights counts {prompts}, {weights}")
        self.type = 'AND'

    def __repr__(self):
        return f"Conjunction:{self.prompts} | weights {self.weights}"
    def __eq__(self, other):
        return type(other) is Conjunction \
               and other.prompts == self.prompts \
               and other.weights == self.weights


class Blend():
    """
    Stores a Blend of multiple Prompts. To apply, build feature vectors for each of the child Prompts and then perform a
    weighted blend of the feature vectors to produce a single feature vector that is effectively a lerp between the
    Prompts.
    """
    def __init__(self, prompts: list, weights: list[float], normalize_weights: bool=True):
        #print("making Blend with prompts", prompts, "and weights", weights)
        weights = [1.0]*len(prompts) if (weights is None or len(weights)==0) else list(weights)
        if len(prompts) != len(weights):
            raise PromptParser.ParsingException(f"while parsing Blend: mismatched prompts/weights counts {prompts}, {weights}")
        for p in prompts:
            if type(p) is not Prompt and type(p) is not FlattenedPrompt:
                raise(PromptParser.ParsingException(f"{type(p)} cannot be added to a Blend, only Prompts or FlattenedPrompts"))
            for f in p.children:
                if isinstance(f, CrossAttentionControlSubstitute):
                    raise(PromptParser.ParsingException(f"while parsing Blend: sorry, you cannot do .swap() as part of a Blend"))

        # upcast all lists to Prompt objects
        self.prompts = [x if (type(x) is Prompt or type(x) is FlattenedPrompt)
                         else Prompt(x)
                        for x in prompts]
        self.prompts = prompts
        self.weights = weights
        self.normalize_weights = normalize_weights

    @property
    def wants_cross_attention_control(self):
        # blends cannot cross-attention control
        return False


    def __repr__(self):
        return f"Blend:{self.prompts} | weights {' ' if self.normalize_weights else '(non-normalized) '}{self.weights}"
    def __eq__(self, other):
        return other.__repr__() == self.__repr__()


class PromptParser():

    class ParsingException(Exception):
        pass

    class UnrecognizedOperatorException(ParsingException):
        def __init__(self, operator:str):
            super().__init__("Unrecognized operator: " + operator)

    def __init__(self, attention_plus_base=1.1, attention_minus_base=0.9):

        self.conjunction, self.prompt = build_parser_syntax(attention_plus_base, attention_minus_base)


    def parse_conjunction(self, prompt: str) -> Conjunction:
        '''
        :param prompt: The prompt string to parse
        :return: a Conjunction representing the parsed results.
        '''
        #print(f"!!parsing '{prompt}'")

        if len(prompt.strip()) == 0:
            return Conjunction(prompts=[FlattenedPrompt([('', 1.0)])], weights=[1.0])

        root = self.conjunction.parse_string(prompt)
        #print(f"'{prompt}' parsed to root", root)
        #fused = fuse_fragments(parts)
        #print("fused to", fused)

        return self.flatten(root[0])

    def parse_legacy_blend(self, text: str, skip_normalize: bool = False) -> Optional[Blend]:
        weighted_subprompts = split_weighted_subprompts(text, skip_normalize=skip_normalize)
        if len(weighted_subprompts) <= 1:
            return None
        strings = [x[0] for x in weighted_subprompts]
        weights = [x[1] for x in weighted_subprompts]

        parsed_conjunctions = [self.parse_conjunction(x) for x in strings]
        flattened_prompts = [x.prompts[0] for x in parsed_conjunctions]

        return Blend(prompts=flattened_prompts, weights=weights, normalize_weights=not skip_normalize)


    def flatten(self, root: Conjunction, verbose = False) -> Conjunction:
        """
        Flattening a Conjunction traverses all of the nested tree-like structures in each of its Prompts or Blends,
        producing from each of these walks a linear sequence of Fragment or CrossAttentionControlSubstitute objects
        that can be readily tokenized without the need to walk a complex tree structure.

        :param root: The Conjunction to flatten.
        :return: A Conjunction containing the result of flattening each of the prompts in the passed-in root.
        """

        def fuse_fragments(items):
            # print("fusing fragments in ", items)
            result = []
            for x in items:
                if type(x) is CrossAttentionControlSubstitute:
                    original_fused = fuse_fragments(x.original)
                    edited_fused = fuse_fragments(x.edited)
                    result.append(CrossAttentionControlSubstitute(original_fused, edited_fused, options=x.options))
                else:
                    last_weight = result[-1].weight \
                        if (len(result) > 0 and not issubclass(type(result[-1]), CrossAttentionControlledFragment)) \
                        else None
                    this_text = x.text
                    this_weight = x.weight
                    if last_weight is not None and last_weight == this_weight:
                        last_text = result[-1].text
                        result[-1] = Fragment(last_text + ' ' + this_text, last_weight)
                    else:
                        result.append(x)
            return result

        def flatten_internal(node, weight_scale, results, prefix):
            verbose and print(prefix + "flattening", node, "...")
            if type(node) is pp.ParseResults or type(node) is list:
                for x in node:
                    results = flatten_internal(x, weight_scale, results, prefix+' pr ')
                #print(prefix, " ParseResults expanded, results is now", results)
            elif type(node) is Attention:
                # if node.weight < 1:
                # todo: inject a blend when flattening attention with weight <1"
                for index,c in enumerate(node.children):
                    results = flatten_internal(c, weight_scale * node.weight, results, prefix + f" att{index} ")
            elif type(node) is Fragment:
                results += [Fragment(node.text, node.weight*weight_scale)]
            elif type(node) is CrossAttentionControlSubstitute:
                original = flatten_internal(node.original, weight_scale, [], prefix + ' CAo ')
                edited = flatten_internal(node.edited, weight_scale, [], prefix + ' CAe ')
                results += [CrossAttentionControlSubstitute(original, edited, options=node.options)]
            elif type(node) is Blend:
                flattened_subprompts = []
                #print(" flattening blend with prompts", node.prompts, "weights", node.weights)
                for prompt in node.prompts:
                    # prompt is a list
                    flattened_subprompts = flatten_internal(prompt, weight_scale, flattened_subprompts, prefix+'B ')
                results += [Blend(prompts=flattened_subprompts, weights=node.weights, normalize_weights=node.normalize_weights)]
            elif type(node) is Prompt:
                #print(prefix + "about to flatten Prompt with children", node.children)
                flattened_prompt = []
                for child in node.children:
                    flattened_prompt = flatten_internal(child, weight_scale, flattened_prompt, prefix+'P ')
                results += [FlattenedPrompt(parts=fuse_fragments(flattened_prompt))]
                #print(prefix + "after flattening Prompt, results is", results)
            else:
                raise PromptParser.ParsingException(f"unhandled node type {type(node)} when flattening {node}")
            verbose and print(prefix + "-> after flattening", type(node).__name__, "results is", results)
            return results

        verbose and print("flattening", root)

        flattened_parts = []
        for part in root.prompts:
            flattened_parts += flatten_internal(part, 1.0, [], ' C| ')

        verbose and print("flattened to", flattened_parts)

        weights = root.weights
        return Conjunction(flattened_parts, weights)




def build_parser_syntax(attention_plus_base: float, attention_minus_base: float):
    def make_operator_object(x):
        #print('making operator for', x)
        target = x[0]
        operator = x[1]
        arguments = x[2]
        if operator == '.attend':
            weight_raw = arguments[0]
            weight = 1.0
            if type(weight_raw) is float or type(weight_raw) is int:
                weight = weight_raw
            elif type(weight_raw) is str:
                base = attention_plus_base if weight_raw[0] == '+' else attention_minus_base
                weight = pow(base, len(weight_raw))
            return Attention(weight=weight, children=[x for x in x[0]])
        elif operator == '.swap':
            return CrossAttentionControlSubstitute(target, arguments, x.as_dict())
        elif operator == '.blend':
            prompts = [Prompt(p) for p in x[0]]
            weights_raw = x[2]
            normalize_weights = True
            if len(weights_raw) > 0 and weights_raw[-1][0] == 'no_normalize':
                normalize_weights = False
                weights_raw = weights_raw[:-1]
            weights = [float(w[0]) for w in weights_raw]
            return Blend(prompts=prompts, weights=weights, normalize_weights=normalize_weights)
        elif operator == '.and' or operator == '.add':
            prompts = [Prompt(p) for p in x[0]]
            weights = [float(w[0]) for w in x[2]]
            return Conjunction(prompts=prompts, weights=weights)

        raise PromptParser.UnrecognizedOperatorException(operator)

    def parse_fragment_str(x, expression: pp.ParseExpression, in_quotes: bool = False, in_parens: bool = False):
        #print(f"parsing fragment string for {x}")
        fragment_string = x[0]
        if len(fragment_string.strip()) == 0:
            return Fragment('')

        if in_quotes:
            # escape unescaped quotes
            fragment_string = fragment_string.replace('"', '\\"')

        try:
            result = (expression + pp.StringEnd()).parse_string(fragment_string)
            #print("parsed to", result)
            return result
        except pp.ParseException as e:
            #print("parse_fragment_str couldn't parse prompt string:", e)
            raise

    # meaningful symbols
    lparen = pp.Literal("(").suppress()
    rparen = pp.Literal(")").suppress()
    quote = pp.Literal('"').suppress()
    comma = pp.Literal(",").suppress()
    dot = pp.Literal(".").suppress()
    equals = pp.Literal("=").suppress()

    escaped_lparen = pp.Literal('\\(')
    escaped_rparen = pp.Literal('\\)')
    escaped_quote = pp.Literal('\\"')
    escaped_comma = pp.Literal('\\,')
    escaped_dot = pp.Literal('\\.')
    escaped_plus = pp.Literal('\\+')
    escaped_minus = pp.Literal('\\-')
    escaped_equals = pp.Literal('\\=')

    syntactic_symbols = {
        '(': escaped_lparen,
        ')': escaped_rparen,
        '"': escaped_quote,
        ',': escaped_comma,
        '.': escaped_dot,
        '+': escaped_plus,
        '-': escaped_minus,
        '=': escaped_equals,
    }
    syntactic_chars = "".join(syntactic_symbols.keys())

    # accepts int or float notation, always maps to float
    number = pp.pyparsing_common.real | \
             pp.Combine(pp.Optional("-")+pp.Word(pp.nums)).set_parse_action(pp.token_map(float))

    # for options
    keyword = pp.Word(pp.alphanums + '_')

    # a word that absolutely does not contain any meaningful syntax
    non_syntax_word = pp.Combine(pp.OneOrMore(pp.MatchFirst([
            pp.Or(syntactic_symbols.values()),
            pp.one_of(['-', '+']) + pp.NotAny(pp.White() | pp.Char(syntactic_chars) | pp.StringEnd()),
            # build character-by-character
            pp.CharsNotIn(string.whitespace + syntactic_chars, exact=1)
        ])))
    non_syntax_word.set_parse_action(lambda x: [Fragment(t) for t in x])
    non_syntax_word.set_name('non_syntax_word')
    non_syntax_word.set_debug(False)

    # a word that can contain any character at all - greedily consumes syntax, so use with care
    free_word = pp.CharsNotIn(string.whitespace).set_parse_action(lambda x: Fragment(x[0]))
    free_word.set_name('free_word')
    free_word.set_debug(False)


    # ok here we go. forward declare some things..
    attention = pp.Forward()
    cross_attention_substitute = pp.Forward()
    parenthesized_fragment = pp.Forward()
    quoted_fragment = pp.Forward()

    # the types of things that can go into a fragment, consisting of syntax-full and/or strictly syntax-free components
    fragment_part_expressions = [
        attention,
        cross_attention_substitute,
        parenthesized_fragment,
        quoted_fragment,
        non_syntax_word
    ]
    # a fragment that is permitted to contain commas
    fragment_including_commas = pp.ZeroOrMore(pp.MatchFirst(
        fragment_part_expressions + [
            pp.Literal(',').set_parse_action(lambda x: Fragment(x[0]))
        ]
    ))
    # a fragment that is not permitted to contain commas
    fragment_excluding_commas = pp.ZeroOrMore(pp.MatchFirst(
        fragment_part_expressions
    ))

    # a fragment in double quotes (may be nested)
    quoted_fragment << pp.QuotedString(quote_char='"', esc_char=None, esc_quote='\\"')
    quoted_fragment.set_parse_action(lambda x: parse_fragment_str(x, fragment_including_commas, in_quotes=True))

    # a fragment inside parentheses (may be nested)
    parenthesized_fragment << (lparen + fragment_including_commas + rparen)
    parenthesized_fragment.set_name('parenthesized_fragment')
    parenthesized_fragment.set_debug(False)

    # a string of the form (<keyword>=<float|keyword> | <float> | <keyword>) where keyword is alphanumeric + '_'
    option = pp.Group(pp.MatchFirst([
        keyword + equals + (number | keyword),  # option=value
        number.copy().set_parse_action(pp.token_map(str)), # weight
        keyword  # flag
    ]))
    # options for an operator, eg "s_start=0.1, 0.3, no_normalize"
    options = pp.Dict(pp.Optional(pp.delimited_list(option)))
    options.set_name('options')
    options.set_debug(False)

    # a fragment which can be used as the target for an operator - either quoted or in parentheses, or a bare vanilla word
    potential_operator_target = (quoted_fragment | parenthesized_fragment | non_syntax_word)

    # a fragment whose weight has been increased or decreased by a given amount
    attention_weight_operator = pp.Word('+') | pp.Word('-') | number
    attention_explicit = (
        pp.Group(potential_operator_target)
        + pp.Literal('.attend')
        + lparen
        + pp.Group(attention_weight_operator)
        + rparen
    )
    attention_explicit.set_parse_action(make_operator_object)
    attention_implicit = (
        pp.Group(potential_operator_target)
        + pp.NotAny(pp.White()) # do not permit whitespace between term and operator
        + pp.Group(attention_weight_operator)
    )
    attention_implicit.set_parse_action(lambda x: make_operator_object([x[0], '.attend', x[1]]))
    attention << (attention_explicit | attention_implicit)
    attention.set_name('attention')
    attention.set_debug(False)

    # cross-attention control by swapping one fragment for another
    cross_attention_substitute << (
        pp.Group(potential_operator_target).set_name('ca-target').set_debug(False)
        + pp.Literal(".swap").set_name('ca-operator').set_debug(False)
        + lparen
        + pp.Group(fragment_excluding_commas).set_name('ca-replacement').set_debug(False)
        + pp.Optional(comma + options).set_name('ca-options').set_debug(False)
        + rparen
    )
    cross_attention_substitute.set_name('cross_attention_substitute')
    cross_attention_substitute.set_debug(False)
    cross_attention_substitute.set_parse_action(make_operator_object)


    # an entire self-contained prompt, which can be used in a Blend or Conjunction
    prompt = pp.ZeroOrMore(pp.MatchFirst([
        cross_attention_substitute,
        attention,
        quoted_fragment,
        parenthesized_fragment,
        free_word,
        pp.White().suppress()
    ]))
    quoted_prompt = quoted_fragment.copy().set_parse_action(lambda x: parse_fragment_str(x, prompt, in_quotes=True))


    # a blend/lerp between the feature vectors for two or more prompts
    blend = (
        lparen
        + pp.Group(pp.delimited_list(pp.Group(potential_operator_target | quoted_prompt), min=1)).set_name('bl-target').set_debug(False)
        + rparen
        + pp.Literal(".blend").set_name('bl-operator').set_debug(False)
        + lparen
        + pp.Group(options).set_name('bl-options').set_debug(False)
        + rparen
    )
    blend.set_name('blend')
    blend.set_debug(False)
    blend.set_parse_action(make_operator_object)

    # an operator to direct stable diffusion to step multiple times, once for each target, and then add the results together with different weights
    explicit_conjunction = (
        lparen
        + pp.Group(pp.delimited_list(pp.Group(potential_operator_target | quoted_prompt), min=1)).set_name('cj-target').set_debug(False)
        + rparen
        + pp.one_of([".and", ".add"]).set_name('cj-operator').set_debug(False)
        + lparen
        + pp.Group(options).set_name('cj-options').set_debug(False)
        + rparen
    )
    explicit_conjunction.set_name('explicit_conjunction')
    explicit_conjunction.set_debug(False)
    explicit_conjunction.set_parse_action(make_operator_object)

    # by default a prompt consists of a Conjunction with a single term
    implicit_conjunction = (blend | pp.Group(prompt)) + pp.StringEnd()
    implicit_conjunction.set_parse_action(lambda x: Conjunction(x))

    conjunction = (explicit_conjunction | implicit_conjunction)

    return conjunction, prompt


def split_weighted_subprompts(text, skip_normalize=False)->list:
    """
    Legacy blend parsing.

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
            "* Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

