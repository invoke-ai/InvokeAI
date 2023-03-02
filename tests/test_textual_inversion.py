
import unittest
from typing import Union

import torch

from invokeai.backend.stable_diffusion import TextualInversionManager


KNOWN_WORDS = ['a', 'b', 'c']
KNOWN_WORDS_TOKEN_IDS = [0, 1, 2]
UNKNOWN_WORDS = ['d', 'e', 'f']

class DummyEmbeddingsList(list):
    def __getattr__(self, name):
        if name == 'num_embeddings':
            return len(self)
        elif name == 'weight':
            return self
        elif name == 'data':
            return self

def make_dummy_embedding():
    return torch.randn([768])

class DummyTransformer:


    def __init__(self):
        self.embeddings = DummyEmbeddingsList([make_dummy_embedding() for _ in range(len(KNOWN_WORDS))])

    def resize_token_embeddings(self, new_size=None):
        if new_size is None:
            return self.embeddings
        else:
            while len(self.embeddings) > new_size:
                self.embeddings.pop(-1)
            while len(self.embeddings) < new_size:
                self.embeddings.append(make_dummy_embedding())

    def get_input_embeddings(self):
        return self.embeddings

class DummyTokenizer():
    def __init__(self):
        self.tokens = KNOWN_WORDS.copy()
        self.bos_token_id = 49406 # these are what the real CLIPTokenizer has
        self.eos_token_id = 49407
        self.pad_token_id = 49407
        self.unk_token_id = 49407

    def convert_tokens_to_ids(self, token_str):
        try:
            return self.tokens.index(token_str)
        except ValueError:
            return self.unk_token_id

    def add_tokens(self, token_str):
        if token_str in self.tokens:
            return 0
        self.tokens.append(token_str)
        return 1


class DummyClipEmbedder:
    def __init__(self):
        self.max_length = 77
        self.transformer = DummyTransformer()
        self.tokenizer = DummyTokenizer()
        self.position_embeddings_tensor = torch.randn([77,768], dtype=torch.float32)

    def position_embedding(self, indices: Union[list,torch.Tensor]):
        if type(indices) is list:
            indices = torch.tensor(indices, dtype=int)
        return torch.index_select(self.position_embeddings_tensor, 0, indices)


def was_embedding_overwritten_correctly(tim: TextualInversionManager, overwritten_embedding: torch.Tensor, ti_indices: list, ti_embedding: torch.Tensor) -> bool:
    return torch.allclose(overwritten_embedding[ti_indices], ti_embedding + tim.clip_embedder.position_embedding(ti_indices))


def make_dummy_textual_inversion_manager():
    return TextualInversionManager(
        tokenizer=DummyTokenizer(),
        text_encoder=DummyTransformer()
    )

class TextualInversionManagerTestCase(unittest.TestCase):


    def test_construction(self):
        tim = make_dummy_textual_inversion_manager()

    def test_add_embedding_for_known_token(self):
        tim = make_dummy_textual_inversion_manager()
        test_embedding = torch.randn([1, 768])
        test_embedding_name = KNOWN_WORDS[0]
        self.assertFalse(tim.has_textual_inversion_for_trigger_string(test_embedding_name))

        pre_embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))

        ti = tim._add_textual_inversion(test_embedding_name, test_embedding)
        self.assertEqual(ti.trigger_token_id, 0)


        # check adding 'test' did not create a new word
        embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count, embeddings_count)

        # check it was added
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name))
        textual_inversion = tim.get_textual_inversion_for_trigger_string(test_embedding_name)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name)
        self.assertEqual(textual_inversion.trigger_token_id, ti.trigger_token_id)

    def test_add_embedding_for_unknown_token(self):
        tim = make_dummy_textual_inversion_manager()
        test_embedding_1 = torch.randn([1, 768])
        test_embedding_name_1 = UNKNOWN_WORDS[0]

        pre_embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))

        added_token_id_1 = tim._add_textual_inversion(test_embedding_name_1, test_embedding_1).trigger_token_id
        # new token id should get added on the end
        self.assertEqual(added_token_id_1, len(KNOWN_WORDS))

        # check adding did create a new word
        embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count+1, embeddings_count)

        # check it was added
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name_1))
        textual_inversion = next(ti for ti in tim.textual_inversions if ti.trigger_token_id == added_token_id_1)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding_1))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name_1)
        self.assertEqual(textual_inversion.trigger_token_id, added_token_id_1)

        # add another one
        test_embedding_2 = torch.randn([1, 768])
        test_embedding_name_2 = UNKNOWN_WORDS[1]

        pre_embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))

        added_token_id_2 = tim._add_textual_inversion(test_embedding_name_2, test_embedding_2).trigger_token_id
        self.assertEqual(added_token_id_2, len(KNOWN_WORDS)+1)

        # check adding did create a new word
        embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count+1, embeddings_count)

        # check it was added
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name_2))
        textual_inversion = next(ti for ti in tim.textual_inversions if ti.trigger_token_id == added_token_id_2)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding_2))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name_2)
        self.assertEqual(textual_inversion.trigger_token_id, added_token_id_2)

        # check the old one is still there
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name_1))
        textual_inversion = next(ti for ti in tim.textual_inversions if ti.trigger_token_id == added_token_id_1)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding_1))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name_1)
        self.assertEqual(textual_inversion.trigger_token_id, added_token_id_1)


    def test_pad_raises_on_eos_bos(self):
        tim = make_dummy_textual_inversion_manager()
        prompt_token_ids_with_eos_bos = [tim.tokenizer.bos_token_id] + \
                                         [KNOWN_WORDS_TOKEN_IDS] + \
                                         [tim.tokenizer.eos_token_id]
        with self.assertRaises(ValueError):
            tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_with_eos_bos)

    def test_pad_tokens_list_vector_length_1(self):
        tim = make_dummy_textual_inversion_manager()
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS.copy()

        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids)
        self.assertEqual(prompt_token_ids, expanded_prompt_token_ids)

        test_embedding_1v = torch.randn([1, 768])
        test_embedding_1v_token = "<inversion-trigger-vector-length-1>"
        test_embedding_1v_token_id = tim._add_textual_inversion(test_embedding_1v_token, test_embedding_1v).trigger_token_id
        self.assertEqual(test_embedding_1v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids_1v_append = prompt_token_ids + [test_embedding_1v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_1v_append)
        self.assertEqual(prompt_token_ids_1v_append, expanded_prompt_token_ids)

        # at the start
        prompt_token_ids_1v_prepend = [test_embedding_1v_token_id] + prompt_token_ids
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_1v_prepend)
        self.assertEqual(prompt_token_ids_1v_prepend, expanded_prompt_token_ids)

        # in the middle
        prompt_token_ids_1v_insert = prompt_token_ids[0:2] + [test_embedding_1v_token_id] + prompt_token_ids[2:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_1v_insert)
        self.assertEqual(prompt_token_ids_1v_insert, expanded_prompt_token_ids)

    def test_pad_tokens_list_vector_length_2(self):
        tim = make_dummy_textual_inversion_manager()
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS.copy()

        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids)
        self.assertEqual(prompt_token_ids, expanded_prompt_token_ids)

        test_embedding_2v = torch.randn([2, 768])
        test_embedding_2v_token = "<inversion-trigger-vector-length-2>"
        test_embedding_2v_token_id = tim._add_textual_inversion(test_embedding_2v_token, test_embedding_2v).trigger_token_id
        test_embedding_2v_pad_token_ids = tim.get_textual_inversion_for_token_id(test_embedding_2v_token_id).pad_token_ids
        self.assertEqual(test_embedding_2v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids_2v_append = prompt_token_ids + [test_embedding_2v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_2v_append)
        self.assertNotEqual(prompt_token_ids_2v_append, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids + [test_embedding_2v_token_id] + test_embedding_2v_pad_token_ids, expanded_prompt_token_ids)

        # at the start
        prompt_token_ids_2v_prepend = [test_embedding_2v_token_id] + prompt_token_ids
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_2v_prepend)
        self.assertNotEqual(prompt_token_ids_2v_prepend, expanded_prompt_token_ids)
        self.assertEqual([test_embedding_2v_token_id] + test_embedding_2v_pad_token_ids + prompt_token_ids, expanded_prompt_token_ids)

        # in the middle
        prompt_token_ids_2v_insert = prompt_token_ids[0:2] + [test_embedding_2v_token_id] + prompt_token_ids[2:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_2v_insert)
        self.assertNotEqual(prompt_token_ids_2v_insert, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids[0:2] + [test_embedding_2v_token_id] + test_embedding_2v_pad_token_ids + prompt_token_ids[2:3], expanded_prompt_token_ids)

    def test_pad_tokens_list_vector_length_8(self):
        tim = make_dummy_textual_inversion_manager()
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS.copy()

        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids)
        self.assertEqual(prompt_token_ids, expanded_prompt_token_ids)

        test_embedding_8v = torch.randn([8, 768])
        test_embedding_8v_token = "<inversion-trigger-vector-length-8>"
        test_embedding_8v_token_id = tim._add_textual_inversion(test_embedding_8v_token, test_embedding_8v).trigger_token_id
        test_embedding_8v_pad_token_ids = tim.get_textual_inversion_for_token_id(test_embedding_8v_token_id).pad_token_ids
        self.assertEqual(test_embedding_8v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids_8v_append = prompt_token_ids + [test_embedding_8v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_8v_append)
        self.assertNotEqual(prompt_token_ids_8v_append, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids + [test_embedding_8v_token_id] + test_embedding_8v_pad_token_ids, expanded_prompt_token_ids)

        # at the start
        prompt_token_ids_8v_prepend = [test_embedding_8v_token_id] + prompt_token_ids
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_8v_prepend)
        self.assertNotEqual(prompt_token_ids_8v_prepend, expanded_prompt_token_ids)
        self.assertEqual([test_embedding_8v_token_id] + test_embedding_8v_pad_token_ids + prompt_token_ids, expanded_prompt_token_ids)

        # in the middle
        prompt_token_ids_8v_insert = prompt_token_ids[0:2] + [test_embedding_8v_token_id] + prompt_token_ids[2:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids_if_necessary(prompt_token_ids=prompt_token_ids_8v_insert)
        self.assertNotEqual(prompt_token_ids_8v_insert, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids[0:2] + [test_embedding_8v_token_id] + test_embedding_8v_pad_token_ids + prompt_token_ids[2:3], expanded_prompt_token_ids)


    def test_deferred_loading(self):
        tim = make_dummy_textual_inversion_manager()
        test_embedding = torch.randn([1, 768])
        test_embedding_name = UNKNOWN_WORDS[0]
        self.assertFalse(tim.has_textual_inversion_for_trigger_string(test_embedding_name))

        pre_embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))

        ti = tim._add_textual_inversion(test_embedding_name, test_embedding, defer_injecting_tokens=True)
        self.assertIsNone(ti.trigger_token_id)

        # check that a new word is not yet created
        embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count, embeddings_count)

        # check it was added
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name))
        textual_inversion = tim.get_textual_inversion_for_trigger_string(test_embedding_name)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name)
        self.assertIsNone(textual_inversion.trigger_token_id, ti.trigger_token_id)

        # check it lazy-loads
        prompt = " ".join([KNOWN_WORDS[0], UNKNOWN_WORDS[0], KNOWN_WORDS[1]])
        tim.create_deferred_token_ids_for_any_trigger_terms(prompt)

        embeddings_count = len(tim.text_encoder.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count+1, embeddings_count)

        textual_inversion = tim.get_textual_inversion_for_trigger_string(test_embedding_name)
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name)
        self.assertEqual(textual_inversion.trigger_token_id, len(KNOWN_WORDS))
