import unittest

from ldm.invoke.prompt_parser import PromptParser, Blend, Conjunction, FlattenedPrompt

def parse_prompt(prompt_string):
    pp = PromptParser()
    #print(f"parsing '{prompt_string}'")
    parse_result = pp.parse(prompt_string)
    #print(f"-> parsed '{prompt_string}' to {parse_result}")
    return parse_result

class PromptParserTestCase(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('', 1)])]), parse_prompt(''))

    def test_basic(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('fire (flames)', 1)])]), parse_prompt("fire (flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([("fire flames", 1)])]), parse_prompt("fire flames"))
        self.assertEqual(Conjunction([FlattenedPrompt([("fire, flames", 1)])]), parse_prompt("fire, flames"))
        self.assertEqual(Conjunction([FlattenedPrompt([("fire, flames , fire", 1)])]), parse_prompt("fire, flames , fire"))

    def test_attention(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('flames', 0.5)])]), parse_prompt("0.5(flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire flames', 0.5)])]), parse_prompt("0.5(fire flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flames', 1.1)])]), parse_prompt("+(flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flames', 0.9)])]), parse_prompt("-(flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1), ('flames', 0.5)])]), parse_prompt("fire 0.5(flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flames', pow(1.1, 2))])]), parse_prompt("++(flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flames', pow(0.9, 2))])]), parse_prompt("--(flames)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flowers', pow(0.9, 3)), ('flames', pow(1.1, 3))])]), parse_prompt("---(flowers) +++flames"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flowers', pow(0.9, 3)), ('flames', pow(1.1, 3))])]), parse_prompt("---(flowers) +++flames"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flowers', pow(0.9, 3)), ('flames+', pow(1.1, 3))])]),
                         parse_prompt("---(flowers) +++flames+"))
        self.assertEqual(Conjunction([FlattenedPrompt([('pretty flowers', 1.1)])]),
                         parse_prompt("+(pretty flowers)"))
        self.assertEqual(Conjunction([FlattenedPrompt([('pretty flowers', 1.1), (', the flames are too hot', 1)])]),
                         parse_prompt("+(pretty flowers), the flames are too hot"))

    def test_no_parens_attention_runon(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', pow(1.1, 2)), ('flames', 1.0)])]), parse_prompt("++fire flames"))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', pow(0.9, 2)), ('flames', 1.0)])]), parse_prompt("--fire flames"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flowers', 1.0), ('fire', pow(1.1, 2)), ('flames', 1.0)])]), parse_prompt("flowers ++fire flames"))
        self.assertEqual(Conjunction([FlattenedPrompt([('flowers', 1.0), ('fire', pow(0.9, 2)), ('flames', 1.0)])]), parse_prompt("flowers --fire flames"))


    def test_explicit_conjunction(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])]), parse_prompt('("fire", "flames").and(1,1)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])]), parse_prompt('("fire", "flames").and()'))
        self.assertEqual(
            Conjunction([FlattenedPrompt([('fire flames', 1.0)]), FlattenedPrompt([('mountain man', 1.0)])]), parse_prompt('("fire flames", "mountain man").and()'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 2.0)]), FlattenedPrompt([('flames', 0.9)])]), parse_prompt('("2.0(fire)", "-flames").and()'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)]),
                                      FlattenedPrompt([('mountain man', 1.0)])]), parse_prompt('("fire", "flames", "mountain man").and()'))

    def test_conjunction_weights(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])], weights=[2.0,1.0]), parse_prompt('("fire", "flames").and(2,1)'))
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0)]), FlattenedPrompt([('flames', 1.0)])], weights=[1.0,2.0]), parse_prompt('("fire", "flames").and(1,2)'))

        with self.assertRaises(PromptParser.ParsingException):
            parse_prompt('("fire", "flames").and(2)')
            parse_prompt('("fire", "flames").and(2,1,2)')

    def test_complex_conjunction(self):
        self.assertEqual(Conjunction([FlattenedPrompt([("mountain man", 1.0)]), FlattenedPrompt([("a person with a hat", 1.0), ("riding a bicycle", pow(1.1,2))])], weights=[0.5, 0.5]),
                         parse_prompt("(\"mountain man\", \"a person with a hat ++(riding a bicycle)\").and(0.5, 0.5)"))

    def test_badly_formed(self):
        def make_untouched_prompt(prompt):
            return Conjunction([FlattenedPrompt([(prompt, 1.0)])])

        def assert_if_prompt_string_not_untouched(prompt):
            self.assertEqual(make_untouched_prompt(prompt), parse_prompt(prompt))

        assert_if_prompt_string_not_untouched('a test prompt')
        assert_if_prompt_string_not_untouched('a badly (formed test prompt')
        assert_if_prompt_string_not_untouched('a badly formed test+ prompt')
        assert_if_prompt_string_not_untouched('a badly (formed test+ prompt')
        assert_if_prompt_string_not_untouched('a badly (formed test+ )prompt')
        assert_if_prompt_string_not_untouched('a badly (formed test+ )prompt')
        assert_if_prompt_string_not_untouched('(((a badly (formed test+ )prompt')
        assert_if_prompt_string_not_untouched('(a (ba)dly (f)ormed test+ prompt')
        self.assertEqual(Conjunction([FlattenedPrompt([('(a (ba)dly (f)ormed test+', 1.0), ('prompt', 1.1)])]),
                         parse_prompt('(a (ba)dly (f)ormed test+ +prompt'))
        self.assertEqual(Conjunction([Blend([FlattenedPrompt([('((a badly (formed test+', 1.0)])], weights=[1.0])]),
                         parse_prompt('("((a badly (formed test+ ").blend(1.0)'))

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
                         parse_prompt("(\"fire\", \"fire flames ++(hot)\", \"hi\").blend(0.7, 0.3, 1.0)")
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


    def test_nested(self):
        self.assertEqual(Conjunction([FlattenedPrompt([('fire', 1.0), ('flames', 2.0), ('trees', 3.0)])]),
                         parse_prompt('fire 2.0(flames 1.5(trees))'))
        self.assertEqual(Conjunction([Blend(prompts=[FlattenedPrompt([('fire', 1.0), ('flames', 1.2100000000000002)]),
                                                     FlattenedPrompt([('mountain', 1.0), ('man', 2.0)])],
                                            weights=[1.0, 1.0])]),
                         parse_prompt('("fire ++(flames)", "mountain 2(man)").blend(1,1)'))

if __name__ == '__main__':
    unittest.main()
