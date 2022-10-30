import unittest

import pyparsing

from ldm.invoke.prompt_parser import PromptParser, Blend, Conjunction, FlattenedPrompt, CrossAttentionControlSubstitute, \
    Fragment


def parse_prompt(prompt_string):
    pp = PromptParser()
    #print(f"parsing '{prompt_string}'")
    parse_result = pp.parse_conjunction(prompt_string)
    #print(f"-> parsed '{prompt_string}' to {parse_result}")
    return parse_result

def make_basic_conjunction(strings: list[str]):
    fragments = [Fragment(x) for x in strings]
    return Conjunction([FlattenedPrompt(fragments)])

def make_weighted_conjunction(weighted_strings: list[tuple[str,float]]):
    fragments = [Fragment(x, w) for x,w in weighted_strings]
    return Conjunction([FlattenedPrompt(fragments)])


class PromptParserTestCase(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(make_weighted_conjunction([('', 1)]), parse_prompt(''))

    def test_basic(self):
        self.assertEqual(make_weighted_conjunction([('fire flames', 1)]), parse_prompt("fire (flames)"))
        self.assertEqual(make_weighted_conjunction([("fire flames", 1)]), parse_prompt("fire flames"))
        self.assertEqual(make_weighted_conjunction([("fire, flames", 1)]), parse_prompt("fire, flames"))
        self.assertEqual(make_weighted_conjunction([("fire, flames , fire", 1)]), parse_prompt("fire, flames , fire"))

    def test_attention(self):
        self.assertEqual(make_weighted_conjunction([('flames', 0.5)]), parse_prompt("(flames)0.5"))
        self.assertEqual(make_weighted_conjunction([('fire flames', 0.5)]), parse_prompt("(fire flames)0.5"))
        self.assertEqual(make_weighted_conjunction([('flames', 1.1)]), parse_prompt("(flames)+"))
        self.assertEqual(make_weighted_conjunction([('flames', 1.1)]), parse_prompt("flames+"))
        self.assertEqual(make_weighted_conjunction([('flames', 1.1)]), parse_prompt("\"flames\"+"))
        self.assertEqual(make_weighted_conjunction([('flames', 0.9)]), parse_prompt("(flames)-"))
        self.assertEqual(make_weighted_conjunction([('flames', 0.9)]), parse_prompt("flames-"))
        self.assertEqual(make_weighted_conjunction([('flames', 0.9)]), parse_prompt("\"flames\"-"))
        self.assertEqual(make_weighted_conjunction([('fire', 1), ('flames', 0.5)]), parse_prompt("fire (flames)0.5"))
        self.assertEqual(make_weighted_conjunction([('flames', pow(1.1, 2))]), parse_prompt("(flames)++"))
        self.assertEqual(make_weighted_conjunction([('flames', pow(0.9, 2))]), parse_prompt("(flames)--"))
        self.assertEqual(make_weighted_conjunction([('flowers', pow(0.9, 3)), ('flames', pow(1.1, 3))]), parse_prompt("(flowers)--- flames+++"))
        self.assertEqual(make_weighted_conjunction([('pretty flowers', 1.1)]),
                         parse_prompt("(pretty flowers)+"))
        self.assertEqual(make_weighted_conjunction([('pretty flowers', 1.1), (', the flames are too hot', 1)]),
                         parse_prompt("(pretty flowers)+, the flames are too hot"))

    def test_no_parens_attention_runon(self):
        self.assertEqual(make_weighted_conjunction([('fire', 1.0), ('flames', pow(1.1, 2))]), parse_prompt("fire flames++"))
        self.assertEqual(make_weighted_conjunction([('fire', 1.0), ('flames', pow(0.9, 2))]), parse_prompt("fire flames--"))
        self.assertEqual(make_weighted_conjunction([('flowers', 1.0), ('fire', pow(1.1, 2)), ('flames', 1.0)]), parse_prompt("flowers fire++ flames"))
        self.assertEqual(make_weighted_conjunction([('flowers', 1.0), ('fire', pow(0.9, 2)), ('flames', 1.0)]), parse_prompt("flowers fire-- flames"))


    def test_explicit_conjunction(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])]), parse_prompt('("fire", "flames").and(1,1)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])]), parse_prompt('("fire", "flames").and()'))
        self.assertEqual(
            Conjunction([FlattenedPrompt([('fire flames', 1.0)]), FlattenedPrompt([('mountain man', 1.0)])]), parse_prompt('("fire flames", "mountain man").and()'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 2.0)]), FlattenedPrompt([('flames', 0.9)])]), parse_prompt('("(fire)2.0", "flames-").and()'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)]),
                                      FlattenedPrompt([('mountain man', 1.0)])]), parse_prompt('("fire", "flames", "mountain man").and()'))

    def test_conjunction_weights(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])], weights=[2.0,1.0]), parse_prompt('("fire", "flames").and(2,1)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])], weights=[1.0,2.0]), parse_prompt('("fire", "flames").and(1,2)'))

        with self.assertRaises(PromptParser.ParsingException):
            parse_prompt('("fire", "flames").and(2)')
            parse_prompt('("fire", "flames").and(2,1,2)')

    def test_complex_conjunction(self):

        #print(parse_prompt("a person with a hat (riding a bicycle.swap(skateboard))++"))

        self.assertEqual(Conjunction([FlattenedPrompt([("mountain man", 1.0)]), FlattenedPrompt([("a person with a hat", 1.0), ("riding a bicycle", pow(1.1,2))])], weights=[0.5, 0.5]),
                         parse_prompt("(\"mountain man\", \"a person with a hat (riding a bicycle)++\").and(0.5, 0.5)"))
        self.assertEqual(Conjunction([FlattenedPrompt([("mountain man", 1.0)]),
                                      FlattenedPrompt([("a person with a hat", 1.0),
                                                       ("riding a", 1.1*1.1),
                                                       CrossAttentionControlSubstitute(
                                                           [Fragment("bicycle", pow(1.1,2))],
                                                           [Fragment("skateboard", pow(1.1,2))])
                                                       ])
                                      ], weights=[0.5, 0.5]),
                         parse_prompt("(\"mountain man\", \"a person with a hat (riding a bicycle.swap(skateboard))++\").and(0.5, 0.5)"))

    def test_badly_formed(self):
        def make_untouched_prompt(prompt):
            return Conjunction([FlattenedPrompt([(prompt, 1.0)])])

        def assert_if_prompt_string_not_untouched(prompt):
            self.assertEqual(make_untouched_prompt(prompt), parse_prompt(prompt))

        assert_if_prompt_string_not_untouched('a test prompt')
        # todo handle this
        #assert_if_prompt_string_not_untouched('a badly formed +test prompt')
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('a badly (formed test prompt')
        #with self.assertRaises(pyparsing.ParseException):
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('a badly (formed +test prompt')
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('a badly (formed +test )prompt')
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('a badly (formed +test )prompt')
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('(((a badly (formed +test )prompt')
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('(a (ba)dly (f)ormed +test prompt')
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('(a (ba)dly (f)ormed +test +prompt')
        with self.assertRaises(pyparsing.ParseException):
            parse_prompt('("((a badly (formed +test ").blend(1.0)')


    def test_blend(self):
        self.assertEqual(Conjunction(
            [Blend([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('fire flames', 1.0)])], [0.7, 0.3])]),
                         parse_prompt("(\"fire\", \"fire flames\").blend(0.7, 0.3)")
                         )
        self.assertEqual(Conjunction([Blend(
            [FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('fire flames', 1.0)]), FlattenedPrompt([('hi', 1.0)])],
            [0.7, 0.3, 1.0])]),
                         parse_prompt("(\"fire\", \"fire flames\", \"hi\").blend(0.7, 0.3, 1.0)")
                         )
        self.assertEqual(Conjunction([Blend([FlattenedPrompt([('fire', 1.0)]),
                                             FlattenedPrompt([('fire flames', 1.0), ('hot', pow(1.1, 2))]),
                                             FlattenedPrompt([('hi', 1.0)])],
                                            weights=[0.7, 0.3, 1.0])]),
                         parse_prompt("(\"fire\", \"fire flames (hot)++\", \"hi\").blend(0.7, 0.3, 1.0)")
                         )
        # blend a single entry is not a failure
        self.assertEqual(Conjunction([Blend([FlattenedPrompt([('fire', 1.0)])], [0.7])]),
                         parse_prompt("(\"fire\").blend(0.7)")
                         )
        # blend with empty
        self.assertEqual(
            Conjunction([Blend([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('', 1.0)])], [0.7, 1.0])]),
            parse_prompt("(\"fire\", \"\").blend(0.7, 1)")
            )
        self.assertEqual(
            Conjunction([Blend([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('', 1.0)])], [0.7, 1.0])]),
            parse_prompt("(\"fire\", \" \").blend(0.7, 1)")
            )
        self.assertEqual(
            Conjunction([Blend([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('', 1.0)])], [0.7, 1.0])]),
            parse_prompt("(\"fire\", \"     \").blend(0.7, 1)")
            )
        self.assertEqual(
            Conjunction([Blend([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([(',', 1.0)])], [0.7, 1.0])]),
            parse_prompt("(\"fire\", \"  ,  \").blend(0.7, 1)")
            )

        self.assertEqual(
            Conjunction([Blend([FlattenedPrompt([('mountain, man, hairy', 1)]),
                                FlattenedPrompt([('face, teeth,', 1), ('eyes', 0.9*0.9)])], weights=[1.0,-1.0])]),
            parse_prompt('("mountain, man, hairy", "face, teeth, eyes--").blend(1,-1)')
        )


    def test_nested(self):
        self.assertEqual(make_weighted_conjunction([('fire', 1.0), ('flames', 2.0), ('trees', 3.0)]),
                         parse_prompt('fire (flames (trees)1.5)2.0'))
        self.assertEqual(Conjunction([Blend(prompts=[FlattenedPrompt([('fire', 1.0), ('flames', 1.2100000000000002)]),
                                                     FlattenedPrompt([('mountain', 1.0), ('man', 2.0)])],
                                            weights=[1.0, 1.0])]),
                         parse_prompt('("fire (flames)++", "mountain (man)2").blend(1,1)'))

    def test_cross_attention_control(self):

        self.assertEqual(Conjunction([
            FlattenedPrompt([Fragment('a', 1),
                             CrossAttentionControlSubstitute([Fragment('cat', 1)], [Fragment('dog', 1)]),
                             Fragment('eating a hotdog', 1)])]), parse_prompt("a \"cat\".swap(dog) eating a hotdog"))

        self.assertEqual(Conjunction([
            FlattenedPrompt([Fragment('a', 1),
                             CrossAttentionControlSubstitute([Fragment('cat', 1)], [Fragment('dog', 1)]),
                             Fragment('eating a hotdog', 1)])]), parse_prompt("a cat.swap(dog) eating a hotdog"))


        fire_flames_to_trees = Conjunction([FlattenedPrompt([('fire', 1.0), \
                                                       CrossAttentionControlSubstitute([Fragment('flames', 1)], [Fragment('trees', 1)])])])
        self.assertEqual(fire_flames_to_trees, parse_prompt('fire "flames".swap(trees)'))
        self.assertEqual(fire_flames_to_trees, parse_prompt('fire (flames).swap(trees)'))
        self.assertEqual(fire_flames_to_trees, parse_prompt('fire ("flames").swap(trees)'))
        self.assertEqual(fire_flames_to_trees, parse_prompt('fire "flames".swap("trees")'))
        self.assertEqual(fire_flames_to_trees, parse_prompt('fire (flames).swap("trees")'))
        self.assertEqual(fire_flames_to_trees, parse_prompt('fire ("flames").swap("trees")'))

        fire_flames_to_trees_and_houses = Conjunction([FlattenedPrompt([('fire', 1.0), \
                                                       CrossAttentionControlSubstitute([Fragment('flames', 1)], [Fragment('trees and houses', 1)])])])
        self.assertEqual(fire_flames_to_trees_and_houses, parse_prompt('fire ("flames").swap("trees and houses")'))
        self.assertEqual(fire_flames_to_trees_and_houses, parse_prompt('fire (flames).swap("trees and houses")'))
        self.assertEqual(fire_flames_to_trees_and_houses, parse_prompt('fire "flames".swap("trees and houses")'))

        trees_and_houses_to_flames = Conjunction([FlattenedPrompt([('fire', 1.0), \
                                                       CrossAttentionControlSubstitute([Fragment('trees and houses', 1)], [Fragment('flames',1)])])])
        self.assertEqual(trees_and_houses_to_flames, parse_prompt('fire ("trees and houses").swap("flames")'))
        self.assertEqual(trees_and_houses_to_flames, parse_prompt('fire (trees and houses).swap("flames")'))
        self.assertEqual(trees_and_houses_to_flames, parse_prompt('fire "trees and houses".swap("flames")'))
        self.assertEqual(trees_and_houses_to_flames, parse_prompt('fire ("trees and houses").swap(flames)'))
        self.assertEqual(trees_and_houses_to_flames, parse_prompt('fire (trees and houses).swap(flames)'))
        self.assertEqual(trees_and_houses_to_flames, parse_prompt('fire "trees and houses".swap(flames)'))

        flames_to_trees_fire = Conjunction([FlattenedPrompt([
                                                       CrossAttentionControlSubstitute([Fragment('flames',1)], [Fragment('trees',1)]),
                                                        (', fire', 1.0)])])
        self.assertEqual(flames_to_trees_fire, parse_prompt('"flames".swap("trees"), fire'))
        self.assertEqual(flames_to_trees_fire, parse_prompt('(flames).swap("trees"), fire'))
        self.assertEqual(flames_to_trees_fire, parse_prompt('("flames").swap("trees"), fire'))
        self.assertEqual(flames_to_trees_fire, parse_prompt('"flames".swap(trees), fire'))
        self.assertEqual(flames_to_trees_fire, parse_prompt('(flames).swap(trees), fire '))
        self.assertEqual(flames_to_trees_fire, parse_prompt('("flames").swap(trees), fire '))


        self.assertEqual(Conjunction([FlattenedPrompt([Fragment('a forest landscape', 1),
                                                                   CrossAttentionControlSubstitute([Fragment('',1)], [Fragment('in winter',1)])])]),
                         parse_prompt('a forest landscape "".swap("in winter")'))
        self.assertEqual(Conjunction([FlattenedPrompt([Fragment('a forest landscape', 1),
                                                                   CrossAttentionControlSubstitute([Fragment('',1)], [Fragment('in winter',1)])])]),
                         parse_prompt('a forest landscape " ".swap("in winter")'))

        self.assertEqual(Conjunction([FlattenedPrompt([Fragment('a forest landscape', 1),
                                                                   CrossAttentionControlSubstitute([Fragment('in winter',1)], [Fragment('',1)])])]),
                         parse_prompt('a forest landscape "in winter".swap("")'))
        self.assertEqual(Conjunction([FlattenedPrompt([Fragment('a forest landscape', 1),
                                                                   CrossAttentionControlSubstitute([Fragment('in winter',1)], [Fragment('',1)])])]),
                         parse_prompt('a forest landscape "in winter".swap()'))
        self.assertEqual(Conjunction([FlattenedPrompt([Fragment('a forest landscape', 1),
                                                                   CrossAttentionControlSubstitute([Fragment('in winter',1)], [Fragment('',1)])])]),
                         parse_prompt('a forest landscape "in winter".swap(" ")'))

    def test_cross_attention_control_with_attention(self):
        flames_to_trees_fire = Conjunction([FlattenedPrompt([
                                                       CrossAttentionControlSubstitute([Fragment('flames',0.5)], [Fragment('trees',0.7)]),
                                                        Fragment(',', 1), Fragment('fire', 2.0)])])
        self.assertEqual(flames_to_trees_fire, parse_prompt('"(flames)0.5".swap("(trees)0.7"), (fire)2.0'))
        flames_to_trees_fire = Conjunction([FlattenedPrompt([
                                                       CrossAttentionControlSubstitute([Fragment('fire',0.5), Fragment('flames',0.25)], [Fragment('trees',0.7)]),
                                                        Fragment(',', 1), Fragment('fire', 2.0)])])
        self.assertEqual(flames_to_trees_fire, parse_prompt('"(fire (flames)0.5)0.5".swap("(trees)0.7"), (fire)2.0'))
        flames_to_trees_fire = Conjunction([FlattenedPrompt([
                                                       CrossAttentionControlSubstitute([Fragment('fire',0.5), Fragment('flames',0.25)], [Fragment('trees',0.7), Fragment('houses', 1)]),
                                                        Fragment(',', 1), Fragment('fire', 2.0)])])
        self.assertEqual(flames_to_trees_fire, parse_prompt('"(fire (flames)0.5)0.5".swap("(trees)0.7 houses"), (fire)2.0'))

    def test_cross_attention_control_options(self):
        self.assertEqual(Conjunction([
            FlattenedPrompt([Fragment('a', 1),
                             CrossAttentionControlSubstitute([Fragment('cat', 1)], [Fragment('dog', 1)], options={'s_start':0.1}),
                             Fragment('eating a hotdog', 1)])]),
            parse_prompt("a \"cat\".swap(dog, s_start=0.1) eating a hotdog"))
        self.assertEqual(Conjunction([
            FlattenedPrompt([Fragment('a', 1),
                             CrossAttentionControlSubstitute([Fragment('cat', 1)], [Fragment('dog', 1)], options={'t_start':0.1}),
                             Fragment('eating a hotdog', 1)])]),
            parse_prompt("a \"cat\".swap(dog, t_start=0.1) eating a hotdog"))
        self.assertEqual(Conjunction([
            FlattenedPrompt([Fragment('a', 1),
                             CrossAttentionControlSubstitute([Fragment('cat', 1)], [Fragment('dog', 1)], options={'s_start': 20.0, 't_start':0.1}),
                             Fragment('eating a hotdog', 1)])]),
            parse_prompt("a \"cat\".swap(dog, t_start=0.1, s_start=20) eating a hotdog"))

        self.assertEqual(
            Conjunction([
                FlattenedPrompt([Fragment('a fantasy forest landscape', 1),
                                 CrossAttentionControlSubstitute([Fragment('', 1)], [Fragment('with a river', 1)],
                                                                 options={'s_start': 0.8, 't_start': 0.8})])]),
            parse_prompt("a fantasy forest landscape \"\".swap(with a river, s_start=0.8, t_start=0.8)"))


    def test_escaping(self):

        # make sure ", ( and ) can be escaped

        self.assertEqual(make_basic_conjunction(['mountain (man)']),parse_prompt('mountain \(man\)'))
        self.assertEqual(make_basic_conjunction(['mountain (man )']),parse_prompt('mountain (\(man)\)'))
        self.assertEqual(make_basic_conjunction(['mountain (man)']),parse_prompt('mountain (\(man\))'))
        self.assertEqual(make_weighted_conjunction([('mountain', 1), ('(man)', 1.1)]), parse_prompt('mountain (\(man\))+'))
        self.assertEqual(make_weighted_conjunction([('mountain', 1), ('(man)', 1.1)]), parse_prompt('"mountain" (\(man\))+'))
        self.assertEqual(make_weighted_conjunction([('"mountain"', 1), ('(man)', 1.1)]), parse_prompt('\\"mountain\\" (\(man\))+'))
        # same weights for each are combined into one
        self.assertEqual(make_weighted_conjunction([('"mountain" (man)', 1.1)]), parse_prompt('(\\"mountain\\")+ (\(man\))+'))
        self.assertEqual(make_weighted_conjunction([('"mountain"', 1.1), ('(man)', 0.9)]), parse_prompt('(\\"mountain\\")+ (\(man\))-'))

        self.assertEqual(make_weighted_conjunction([('mountain', 1), ('\(man\)', 1.1)]),parse_prompt('mountain (\(man\))1.1'))
        self.assertEqual(make_weighted_conjunction([('mountain', 1), ('\(man\)', 1.1)]),parse_prompt('"mountain" (\(man\))1.1'))
        self.assertEqual(make_weighted_conjunction([('"mountain"', 1), ('\(man\)', 1.1)]),parse_prompt('\\"mountain\\" (\(man\))1.1'))
        # same weights for each are combined into one
        self.assertEqual(make_weighted_conjunction([('\\"mountain\\" \(man\)', 1.1)]),parse_prompt('(\\"mountain\\")+ (\(man\))1.1'))
        self.assertEqual(make_weighted_conjunction([('\\"mountain\\"', 1.1), ('\(man\)', 0.9)]),parse_prompt('(\\"mountain\\")1.1 (\(man\))0.9'))

        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain', 1.1), ('\(man\)', 1.1*1.1)]),parse_prompt('hairy (mountain (\(man\))+)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('\(man\)', 1.1*1.1), ('mountain', 1.1)]),parse_prompt('hairy ((\(man\))1.1 "mountain")+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain', 1.1), ('\(man\)', 1.1*1.1)]),parse_prompt('hairy ("mountain" (\(man\))1.1 )+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, man', 1.1)]),parse_prompt('hairy ("mountain, man")+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, man with a', 1.1), ('beard', 1.1*1.1)]), parse_prompt('hairy ("mountain, man" with a beard+)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, man with a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, man" with a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, \"man\" with a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, \\"man\\"" with a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, m\"an\" with a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, m\\"an\\"" with a (beard)2.0)+'))

        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, \"man (with a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, \\\"man\" \(with a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, \"man w(ith a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, \\\"man\" w\(ith a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, \"man with( a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, \\\"man\" with\( a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, \"man )with a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, \\\"man\" \)with a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, \"man w)ith a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, \\\"man\" w\)ith a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mountain, \"man with) a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mountain, \\\"man\" with\) a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy', 1), ('mou)ntain, \"man (wit(h a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy ("mou\)ntain, \\\"man\" \(wit\(h a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hai(ry', 1), ('mountain, \"man w)ith a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hai\(ry ("mountain, \\\"man\" w\)ith a (beard)2.0)+'))
        self.assertEqual(make_weighted_conjunction([('hairy((', 1), ('mountain, \"man with a', 1.1), ('beard', 1.1*2.0)]), parse_prompt('hairy\(\( ("mountain, \\\"man\" with a (beard)2.0)+'))

        self.assertEqual(make_weighted_conjunction([('mountain, \"man (with a', 1.1), ('beard', 1.1*2.0), ('hairy', 1)]), parse_prompt('("mountain, \\\"man\" \(with a (beard)2.0)+ hairy'))
        self.assertEqual(make_weighted_conjunction([('mountain, \"man w(ith a', 1.1), ('beard', 1.1*2.0), ('hairy', 1)]), parse_prompt('("mountain, \\\"man\" w\(ith a (beard)2.0)+hairy'))
        self.assertEqual(make_weighted_conjunction([('mountain, \"man with( a', 1.1), ('beard', 1.1*2.0), ('hairy', 1)]), parse_prompt('("mountain, \\\"man\" with\( a (beard)2.0)+ hairy'))
        self.assertEqual(make_weighted_conjunction([('mountain, \"man )with a', 1.1), ('beard', 1.1*2.0), ('hairy', 1)]), parse_prompt('("mountain, \\\"man\" \)with a (beard)2.0)+ hairy'))
        self.assertEqual(make_weighted_conjunction([('mountain, \"man w)ith a', 1.1), ('beard', 1.1*2.0), ('hairy', 1)]), parse_prompt('("mountain, \\\"man\" w\)ith a (beard)2.0)+ hairy'))
        self.assertEqual(make_weighted_conjunction([('mountain, \"man with) a', 1.1), ('beard', 1.1*2.0), ('hairy', 1)]), parse_prompt(' ("mountain, \\\"man\" with\) a (beard)2.0)+ hairy'))
        self.assertEqual(make_weighted_conjunction([('mou)ntain, \"man (wit(h a', 1.1), ('beard', 1.1*2.0), ('hairy', 1)]), parse_prompt('("mou\)ntain, \\\"man\" \(wit\(h a (beard)2.0)+ hairy'))
        self.assertEqual(make_weighted_conjunction([('mountain, \"man w)ith a', 1.1), ('beard', 1.1*2.0), ('hai(ry', 1)]), parse_prompt('("mountain, \\\"man\" w\)ith a (beard)2.0)+ hai\(ry '))
        self.assertEqual(make_weighted_conjunction([('mountain, \"man with a', 1.1), ('beard', 1.1*2.0), ('hairy((', 1)]), parse_prompt('("mountain, \\\"man\" with a (beard)2.0)+ hairy\(\( '))

    def test_cross_attention_escaping(self):

        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('man', 1)], [Fragment('monkey', 1)])])]),
                         parse_prompt('mountain (man).swap(monkey)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('man', 1)], [Fragment('m(onkey', 1)])])]),
                         parse_prompt('mountain (man).swap(m\(onkey)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('m(an', 1)], [Fragment('m(onkey', 1)])])]),
                         parse_prompt('mountain (m\(an).swap(m\(onkey)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('(((', 1)], [Fragment('m(on))key', 1)])])]),
                         parse_prompt('mountain (\(\(\().swap(m\(on\)\)key)'))

        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('man', 1)], [Fragment('monkey', 1)])])]),
                         parse_prompt('mountain ("man").swap(monkey)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('man', 1)], [Fragment('monkey', 1)])])]),
                         parse_prompt('mountain ("man").swap("monkey")'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('"man', 1)], [Fragment('monkey', 1)])])]),
                         parse_prompt('mountain (\\"man).swap("monkey")'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('man', 1)], [Fragment('m(onkey', 1)])])]),
                         parse_prompt('mountain (man).swap(m\(onkey)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('m(an', 1)], [Fragment('m(onkey', 1)])])]),
                         parse_prompt('mountain (m\(an).swap(m\(onkey)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('mountain', 1), CrossAttentionControlSubstitute([Fragment('(((', 1)], [Fragment('m(on))key', 1)])])]),
                         parse_prompt('mountain (\(\(\().swap(m\(on\)\)key)'))

    def test_legacy_blend(self):
        pp = PromptParser()

        self.assertEqual(Blend([FlattenedPrompt([('mountain man', 1)]),
                                FlattenedPrompt([('man mountain', 1)])],
                                weights=[0.5,0.5]),
                         pp.parse_legacy_blend('mountain man:1 man mountain:1'))

        self.assertEqual(Blend([FlattenedPrompt([('mountain', 1.1), ('man', 1)]),
                                FlattenedPrompt([('man', 1), ('mountain', 0.9)])],
                                weights=[0.5,0.5]),
                         pp.parse_legacy_blend('mountain+ man:1 man mountain-:1'))

        self.assertEqual(Blend([FlattenedPrompt([('mountain', 1.1), ('man', 1)]),
                                FlattenedPrompt([('man', 1), ('mountain', 0.9)])],
                                weights=[0.5,0.5]),
                         pp.parse_legacy_blend('mountain+ man:1 man mountain-'))

        self.assertEqual(Blend([FlattenedPrompt([('mountain', 1.1), ('man', 1)]),
                                FlattenedPrompt([('man', 1), ('mountain', 0.9)])],
                                weights=[0.5,0.5]),
                         pp.parse_legacy_blend('mountain+ man: man mountain-:'))

        self.assertEqual(Blend([FlattenedPrompt([('mountain man', 1)]),
                                FlattenedPrompt([('man mountain', 1)])],
                                weights=[0.75,0.25]),
                         pp.parse_legacy_blend('mountain man:3 man mountain:1'))

        self.assertEqual(Blend([FlattenedPrompt([('mountain man', 1)]),
                                FlattenedPrompt([('man mountain', 1)])],
                                weights=[1.0,0.0]),
                         pp.parse_legacy_blend('mountain man:3 man mountain:0'))

        self.assertEqual(Blend([FlattenedPrompt([('mountain man', 1)]),
                                FlattenedPrompt([('man mountain', 1)])],
                                weights=[0.8,0.2]),
                         pp.parse_legacy_blend('"mountain man":4 man mountain'))


    def test_single(self):
        # todo handle this
        #self.assertEqual(make_basic_conjunction(['a badly formed +test prompt']),
        #                 parse_prompt('a badly formed +test prompt'))
        pass


if __name__ == '__main__':
    unittest.main()
