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
                raise PromptParser.ParsingException(f"Prompt cannot contain {type(c).__name__} {c}, only {BaseFragment.__subclasses__()} are allowed")
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
        self.weight = weight
        self.children = children
        #print(f"A: requested attention '{children}' to {weight}")

    def __repr__(self):
        return f"Attention:'{self.children}' @ {self.weight}"
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
    def __init__(self, original: Union[Fragment, list], edited: Union[Fragment, list], options: dict=None):
        self.original = original
        self.edited = edited

        default_options = {
            's_start': 0.0,
            's_end': 0.2062994740159002, # ~= shape_freedom=0.5
            't_start': 0.0,
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
        #print("making conjunction with", parts)
        self.prompts = [x if (type(x) is Prompt
                          or type(x) is Blend
                          or type(x) is FlattenedPrompt)
                      else Prompt(x) for x in prompts]
        self.weights = [1.0]*len(self.prompts) if weights is None else list(weights)
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

    def __repr__(self):
        return f"Blend:{self.prompts} | weights {' ' if self.normalize_weights else '(non-normalized) '}{self.weights}"
    def __eq__(self, other):
        return other.__repr__() == self.__repr__()


class PromptParser():

    class ParsingException(Exception):
        pass

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

    def parse_legacy_blend(self, text: str) -> Optional[Blend]:
        weighted_subprompts = split_weighted_subprompts(text, skip_normalize=False)
        if len(weighted_subprompts) <= 1:
            return None
        strings = [x[0] for x in weighted_subprompts]
        weights = [x[1] for x in weighted_subprompts]

        parsed_conjunctions = [self.parse_conjunction(x) for x in strings]
        flattened_prompts = [x.prompts[0] for x in parsed_conjunctions]

        return Blend(prompts=flattened_prompts, weights=weights, normalize_weights=True)


    def flatten(self, root: Conjunction) -> Conjunction:
        """
        Flattening a Conjunction traverses all of the nested tree-like structures in each of its Prompts or Blends,
        producing from each of these walks a linear sequence of Fragment or CrossAttentionControlSubstitute objects
        that can be readily tokenized without the need to walk a complex tree structure.

        :param root: The Conjunction to flatten.
        :return: A Conjunction containing the result of flattening each of the prompts in the passed-in root.
        """

        #print("flattening", root)

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
            #print(prefix + "flattening", node, "...")
            if type(node) is pp.ParseResults:
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
            #print(prefix + "-> after flattening", type(node).__name__, "results is", results)
            return results


        flattened_parts = []
        for part in root.prompts:
            flattened_parts += flatten_internal(part, 1.0, [], ' C| ')

        #print("flattened to", flattened_parts)

        weights = root.weights
        return Conjunction(flattened_parts, weights)



def build_parser_syntax(attention_plus_base: float, attention_minus_base: float):

    lparen = pp.Literal("(").suppress()
    rparen = pp.Literal(")").suppress()
    quotes = pp.Literal('"').suppress()
    comma = pp.Literal(",").suppress()

    # accepts int or float notation, always maps to float
    number = pp.pyparsing_common.real | \
             pp.Combine(pp.Optional("-")+pp.Word(pp.nums)).set_parse_action(pp.token_map(float))

    attention = pp.Forward()
    quoted_fragment = pp.Forward()
    parenthesized_fragment = pp.Forward()
    cross_attention_substitute = pp.Forward()

    def make_text_fragment(x):
        #print("### making fragment for", x)
        if type(x[0]) is Fragment:
            assert(False)
        if type(x) is str:
            return Fragment(x)
        elif type(x) is pp.ParseResults or type(x) is list:
            #print(f'converting {type(x).__name__} to Fragment')
            return Fragment(' '.join([s for s in x]))
        else:
            raise PromptParser.ParsingException("Cannot make fragment from " + str(x))

    def build_escaped_word_parser_charbychar(escaped_chars_to_ignore: str):
        escapes = []
        for c in escaped_chars_to_ignore:
            escapes.append(pp.Literal('\\'+c))
        return pp.Combine(pp.OneOrMore(
            pp.MatchFirst(escapes + [pp.CharsNotIn(
                string.whitespace + escaped_chars_to_ignore,
                exact=1
            )])
        ))



    def parse_fragment_str(x, in_quotes: bool=False, in_parens: bool=False):
        #print(f"parsing fragment string for {x}")
        fragment_string = x[0]
        #print(f"ppparsing fragment string \"{fragment_string}\"")

        if len(fragment_string.strip()) == 0:
            return Fragment('')

        if in_quotes:
            # escape unescaped quotes
            fragment_string = fragment_string.replace('"', '\\"')

        #fragment_parser = pp.Group(pp.OneOrMore(attention | cross_attention_substitute | (greedy_word.set_parse_action(make_text_fragment))))
        try:
            result = pp.Group(pp.MatchFirst([
                    pp.OneOrMore(quoted_fragment | attention | unquoted_word).set_name('pf_str_qfuq'),
                    pp.Empty().set_parse_action(make_text_fragment) + pp.StringEnd()
            ])).set_name('blend-result').set_debug(False).parse_string(fragment_string)
            #print("parsed to", result)
            return result
        except pp.ParseException as e:
            #print("parse_fragment_str couldn't parse prompt string:", e)
            raise

    quoted_fragment << pp.QuotedString(quote_char='"', esc_char=None, esc_quote='\\"')
    quoted_fragment.set_parse_action(lambda x: parse_fragment_str(x, in_quotes=True)).set_name('quoted_fragment')

    escaped_quote = pp.Literal('\\"')#.set_parse_action(lambda x: '"')
    escaped_lparen = pp.Literal('\\(')#.set_parse_action(lambda x: '(')
    escaped_rparen = pp.Literal('\\)')#.set_parse_action(lambda x: ')')
    escaped_backslash = pp.Literal('\\\\')#.set_parse_action(lambda x: '"')

    empty = (
            (lparen + pp.ZeroOrMore(pp.Word(string.whitespace)) + rparen) |
            (quotes + pp.ZeroOrMore(pp.Word(string.whitespace)) + quotes)).set_debug(False).set_name('empty')


    def not_ends_with_swap(x):
        #print("trying to match:", x)
        return not x[0].endswith('.swap')

    unquoted_word = (pp.Combine(pp.OneOrMore(
            escaped_rparen | escaped_lparen | escaped_quote | escaped_backslash |
            (pp.CharsNotIn(string.whitespace + '\\"()', exact=1)
    )))
            # don't whitespace when the next word starts with +, eg "badly +formed"
         + (pp.White().suppress() |
            # don't eat +/-
            pp.NotAny(pp.Word('+') | pp.Word('-'))
            )
                     )

    unquoted_word.set_parse_action(make_text_fragment).set_name('unquoted_word').set_debug(False)
    #print(unquoted_fragment.parse_string("cat.swap(dog)"))

    parenthesized_fragment << (lparen +
       pp.Or([
        (parenthesized_fragment),
        (quoted_fragment.copy().set_parse_action(lambda x: parse_fragment_str(x, in_quotes=True)).set_debug(False)).set_name('-quoted_paren_internal').set_debug(False),
        (pp.Combine(pp.OneOrMore(
            escaped_quote | escaped_lparen | escaped_rparen | escaped_backslash |
            pp.CharsNotIn(string.whitespace + '\\"()', exact=1) |
            pp.White()
        )).set_name('--combined').set_parse_action(lambda x: parse_fragment_str(x, in_parens=True)).set_debug(False)),
        pp.Empty()
       ]) + rparen)
    parenthesized_fragment.set_name('parenthesized_fragment').set_debug(False)

    debug_attention = False
    # attention control of the form (phrase)+ / (phrase)+ / (phrase)<weight>
    # phrase can be multiple words, can have multiple +/- signs to increase the effect or type a floating point or integer weight
    attention_with_parens = pp.Forward()
    attention_without_parens = pp.Forward()

    attention_with_parens_foot = (number | pp.Word('+') | pp.Word('-'))\
        .set_name("attention_foot")\
        .set_debug(False)
    attention_with_parens <<= pp.Group(
        lparen +
        pp.ZeroOrMore(quoted_fragment | attention_with_parens | parenthesized_fragment | cross_attention_substitute | attention_without_parens |
                      (pp.Empty() + build_escaped_word_parser_charbychar('()')).set_name('undecorated_word').set_debug(debug_attention)#.set_parse_action(lambda t: t[0])
                  )
        + rparen + attention_with_parens_foot)
    attention_with_parens.set_name('attention_with_parens').set_debug(debug_attention)

    attention_without_parens_foot = (pp.NotAny(pp.White()) + pp.Or([pp.Word('+'), pp.Word('-')]) + pp.FollowedBy(pp.StringEnd() | pp.White() | pp.Literal('(') | pp.Literal(')') | pp.Literal(',') | pp.Literal('"')) ).set_name('attention_without_parens_foots')
    attention_without_parens <<= pp.Group(pp.MatchFirst([
        quoted_fragment.copy().set_name('attention_quoted_fragment_without_parens').set_debug(debug_attention) + attention_without_parens_foot,
        pp.Combine(build_escaped_word_parser_charbychar('()+-')).set_name('attention_word_without_parens').set_debug(debug_attention)#.set_parse_action(lambda x: print('escapéd', x))
                                 + attention_without_parens_foot#.leave_whitespace()
    ]))
    attention_without_parens.set_name('attention_without_parens').set_debug(debug_attention)


    attention << pp.MatchFirst([attention_with_parens,
                  attention_without_parens
                  ])
    attention.set_name('attention')

    def make_attention(x):
        #print("entered make_attention with", x)
        children = x[0][:-1]
        weight_raw = x[0][-1]
        weight = 1.0
        if type(weight_raw) is float or type(weight_raw) is int:
            weight = weight_raw
        elif type(weight_raw) is str:
            base = attention_plus_base if weight_raw[0] == '+' else attention_minus_base
            weight = pow(base, len(weight_raw))

        #print("making Attention from", children, "with weight", weight)

        return Attention(weight=weight, children=[(Fragment(x) if type(x) is str else x) for x in children])

    attention_with_parens.set_parse_action(make_attention)
    attention_without_parens.set_parse_action(make_attention)

    #print("parsing test:", attention_with_parens.parse_string("mountain (man)1.1"))

    # cross-attention control
    empty_string = ((lparen + rparen) |
                    pp.Literal('""').suppress() |
                    (lparen + pp.Literal('""').suppress() + rparen)
                    ).set_parse_action(lambda x: Fragment(""))
    empty_string.set_name('empty_string')

    # cross attention control
    debug_cross_attention_control = False
    original_fragment = pp.MatchFirst([
                        quoted_fragment.set_debug(debug_cross_attention_control),
                        parenthesized_fragment.set_debug(debug_cross_attention_control),
                        pp.Combine(pp.OneOrMore(pp.CharsNotIn(string.whitespace + '.', exact=1))).set_parse_action(make_text_fragment) + pp.FollowedBy(".swap"),
                        empty_string.set_debug(debug_cross_attention_control),
               ])
    # support keyword=number arguments
    cross_attention_option_keyword = pp.Or([pp.Keyword("s_start"), pp.Keyword("s_end"), pp.Keyword("t_start"), pp.Keyword("t_end"), pp.Keyword("shape_freedom")])
    cross_attention_option = pp.Group(cross_attention_option_keyword + pp.Literal("=").suppress() + number)
    edited_fragment = pp.MatchFirst([
        (lparen + rparen).set_parse_action(lambda x: Fragment('')),
        lparen +
            (quoted_fragment | attention |
                pp.Group(pp.ZeroOrMore(build_escaped_word_parser_charbychar(',)').set_parse_action(make_text_fragment)))
            ) +
            pp.Dict(pp.ZeroOrMore(comma + cross_attention_option)) +
        rparen,
        parenthesized_fragment
    ])
    cross_attention_substitute << original_fragment + pp.Literal(".swap").set_debug(False).suppress() + edited_fragment

    original_fragment.set_name('original_fragment').set_debug(debug_cross_attention_control)
    edited_fragment.set_name('edited_fragment').set_debug(debug_cross_attention_control)
    cross_attention_substitute.set_name('cross_attention_substitute').set_debug(debug_cross_attention_control)

    def make_cross_attention_substitute(x):
        #print("making cacs for", x[0], "->", x[1], "with options", x.as_dict())
        #if len(x>2):
        cacs = CrossAttentionControlSubstitute(x[0], x[1], options=x.as_dict())
        #print("made", cacs)
        return cacs
    cross_attention_substitute.set_parse_action(make_cross_attention_substitute)


    # root prompt definition
    debug_root_prompt = False
    prompt = (pp.OneOrMore(pp.MatchFirst([cross_attention_substitute.set_debug(debug_root_prompt),
                                  attention.set_debug(debug_root_prompt),
                                  quoted_fragment.set_debug(debug_root_prompt),
                                  parenthesized_fragment.set_debug(debug_root_prompt),
                                  unquoted_word.set_debug(debug_root_prompt),
                                  empty.set_parse_action(make_text_fragment).set_debug(debug_root_prompt)])
                           ) + pp.StringEnd()) \
        .set_name('prompt') \
        .set_parse_action(lambda x: Prompt(x)) \
        .set_debug(debug_root_prompt)

    #print("parsing test:", prompt.parse_string("spaced eyes--"))
    #print("parsing test:", prompt.parse_string("eyes--"))

    # weighted blend of prompts
    # ("promptA", "promptB").blend(a, b) where "promptA" and "promptB" are valid prompts and a and b are float or
    # int weights.
    # can specify more terms eg ("promptA", "promptB", "promptC").blend(a,b,c)

    def make_prompt_from_quoted_string(x):
        #print(' got quoted prompt', x)

        x_unquoted = x[0][1:-1]
        if len(x_unquoted.strip()) == 0:
            # print(' b : just an empty string')
            return Prompt([Fragment('')])
        #print(f' b parsing \'{x_unquoted}\'')
        x_parsed = prompt.parse_string(x_unquoted)
        #print(" quoted prompt was parsed to", type(x_parsed),":", x_parsed)
        return x_parsed[0]

    quoted_prompt = pp.dbl_quoted_string.set_parse_action(make_prompt_from_quoted_string)
    quoted_prompt.set_name('quoted_prompt')

    debug_blend=False
    blend_terms = pp.delimited_list(quoted_prompt).set_name('blend_terms').set_debug(debug_blend)
    blend_weights = (pp.delimited_list(number) + pp.Optional(pp.Char(",").suppress() + "no_normalize")).set_name('blend_weights').set_debug(debug_blend)
    blend = pp.Group(lparen + pp.Group(blend_terms) + rparen
                     + pp.Literal(".blend").suppress()
                     + lparen + pp.Group(blend_weights) + rparen).set_name('blend')
    blend.set_debug(debug_blend)

    def make_blend(x):
        prompts = x[0][0]
        weights = x[0][1]
        normalize = True
        if weights[-1] == 'no_normalize':
            normalize = False
            weights = weights[:-1]
        return Blend(prompts=prompts, weights=weights, normalize_weights=normalize)

    blend.set_parse_action(make_blend)

    conjunction_terms = blend_terms.copy().set_name('conjunction_terms')
    conjunction_weights = blend_weights.copy().set_name('conjunction_weights')
    conjunction_with_parens_and_quotes = pp.Group(lparen + pp.Group(conjunction_terms) + rparen
                     + pp.Literal(".and").suppress()
                     + lparen + pp.Optional(pp.Group(conjunction_weights)) + rparen).set_name('conjunction')
    def make_conjunction(x):
        parts_raw = x[0][0]
        weights = x[0][1] if len(x[0])>1 else [1.0]*len(parts_raw)
        parts = [part for part in parts_raw]
        return Conjunction(parts, weights)
    conjunction_with_parens_and_quotes.set_parse_action(make_conjunction)

    implicit_conjunction = pp.OneOrMore(blend | prompt).set_name('implicit_conjunction')
    implicit_conjunction.set_parse_action(lambda x: Conjunction(x))

    conjunction = conjunction_with_parens_and_quotes | implicit_conjunction
    conjunction.set_debug(False)

    # top-level is a conjunction of one or more blends or prompts
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
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]


# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, display_label=None):
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', 'x` ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(f"\n>> Tokens {display_label or ''} ({usedTokens}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m"
        )
