
import unittest

import torch

from ldm.modules.embedding_manager import TextualInversionManager


class DummyClipEmbedder:
    max_length = 77
    bos_token_id = 49406
    eos_token_id = 49407

class TextualInversionManagerTestCase(unittest.TestCase):


    def test_construction(self):
        tim = TextualInversionManager(DummyClipEmbedder())

    def test_add_embedding(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        test_embedding = torch.random([1, 768])
        test_embedding_name = "test"
        token_id = tim.add_textual_inversion(test_embedding_name, test_embedding)
        self.assertTrue(tim.has_textual_inversion(test_embedding_name))

        textual_inversion = next(ti for ti in tim.textual_inversions if ti.token_id == token_id)
        self.assertIsNotNone(textual_inversion)
        self.assertEqual(textual_inversion.embedding, test_embedding)
        self.assertEqual(textual_inversion.token_string, test_embedding_name)
        self.assertEqual(textual_inversion.token_id, token_id)

    def test_pad_tokens_list(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        prompt_token_ids = [DummyClipEmbedder.bos_token_id, 0, 1, 2, DummyClipEmbedder.eos_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids)
        self.assertEqual(prompt_token_ids, expanded_prompt_token_ids)

        test_embedding = torch.random([1, 768])
        test_embedding_name = "test"
        tim.add_textual_inversion("<token>",

        self.assertRaises()
