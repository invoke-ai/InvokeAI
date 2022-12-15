
import unittest

import torch

from ldm.modules.embedding_manager import TextualInversionManager


KNOWN_WORDS = ['a', 'b', 'c']
KNOWN_WORDS_TOKEN_IDS = [0, 1, 2]
UNKNOWN_WORDS = ['d', 'e', 'f']

class DummyEmbeddingsList(list):
    def __getattr__(self, name):
        if name == 'num_embeddings':
            return len(self)

class DummyTransformer:
    def __init__(self):
        self.embeddings = DummyEmbeddingsList([0] * len(KNOWN_WORDS))

    def resize_token_embeddings(self, new_size=None):
        if new_size is None:
            return self.embeddings
        else:
            while len(self.embeddings) > new_size:
                self.embeddings.pop(-1)
            while len(self.embeddings) < new_size:
                self.embeddings.append(0)


class DummyTokenizer():
    def __init__(self):
        self.tokens = KNOWN_WORDS.copy()
        self.bos_token_id = 49406
        self.eos_token_id = 49407
        self.pad_token_id = 49407
        self.unk_token_id = 49407

    def convert_tokens_to_ids(self, token_str):
        try:
            return self.tokens.index(token_str)
        except ValueError:
            return self.unk_token_id

    def add_tokens(self, token_str):
        self.tokens.append(token_str)
        return 1


class DummyClipEmbedder:
    def __init__(self):
        self.max_length = 77
        self.transformer = DummyTransformer()
        self.tokenizer = DummyTokenizer()


class TextualInversionManagerTestCase(unittest.TestCase):


    def test_construction(self):
        tim = TextualInversionManager(DummyClipEmbedder())

    def test_add_embedding_for_known_token(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        test_embedding = torch.randn([1, 768])
        test_embedding_name = KNOWN_WORDS[0]
        self.assertFalse(tim.has_textual_inversion_for_trigger_string(test_embedding_name))

        pre_embeddings_count = len(tim.clip_embedder.transformer.resize_token_embeddings(None))

        token_id = tim.add_textual_inversion(test_embedding_name, test_embedding)
        self.assertEqual(token_id, 0)


        # check adding 'test' did not create a new word
        embeddings_count = len(tim.clip_embedder.transformer.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count, embeddings_count)

        # check it was added
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name))
        textual_inversion = tim.get_textual_inversion_for_trigger_string(test_embedding_name)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name)
        self.assertEqual(textual_inversion.token_id, token_id)

    def test_add_embedding_for_unknown_token(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        test_embedding_1 = torch.randn([1, 768])
        test_embedding_name_1 = UNKNOWN_WORDS[0]

        pre_embeddings_count = len(tim.clip_embedder.transformer.resize_token_embeddings(None))

        added_token_id_1 = tim.add_textual_inversion(test_embedding_name_1, test_embedding_1)
        # new token id should get added on the end
        self.assertEqual(added_token_id_1, len(KNOWN_WORDS))

        # check adding did create a new word
        embeddings_count = len(tim.clip_embedder.transformer.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count+1, embeddings_count)

        # check it was added
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name_1))
        textual_inversion = next(ti for ti in tim.textual_inversions if ti.token_id == added_token_id_1)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding_1))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name_1)
        self.assertEqual(textual_inversion.token_id, added_token_id_1)

        # add another one
        test_embedding_2 = torch.randn([1, 768])
        test_embedding_name_2 = UNKNOWN_WORDS[1]

        pre_embeddings_count = len(tim.clip_embedder.transformer.resize_token_embeddings(None))

        added_token_id_2 = tim.add_textual_inversion(test_embedding_name_2, test_embedding_2)
        self.assertEqual(added_token_id_2, len(KNOWN_WORDS)+1)

        # check adding did create a new word
        embeddings_count = len(tim.clip_embedder.transformer.resize_token_embeddings(None))
        self.assertEqual(pre_embeddings_count+1, embeddings_count)

        # check it was added
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name_2))
        textual_inversion = next(ti for ti in tim.textual_inversions if ti.token_id == added_token_id_2)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding_2))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name_2)
        self.assertEqual(textual_inversion.token_id, added_token_id_2)

        # check the old one is still there
        self.assertTrue(tim.has_textual_inversion_for_trigger_string(test_embedding_name_1))
        textual_inversion = next(ti for ti in tim.textual_inversions if ti.token_id == added_token_id_1)
        self.assertIsNotNone(textual_inversion)
        self.assertTrue(torch.equal(textual_inversion.embedding, test_embedding_1))
        self.assertEqual(textual_inversion.trigger_string, test_embedding_name_1)
        self.assertEqual(textual_inversion.token_id, added_token_id_1)


    def test_pad_raises_on_eos_bos(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        prompt_token_ids_with_eos_bos = [tim.clip_embedder.tokenizer.bos_token_id] + \
                                         [KNOWN_WORDS_TOKEN_IDS] + \
                                         [tim.clip_embedder.tokenizer.eos_token_id]
        with self.assertRaises(ValueError):
            expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_with_eos_bos)

    def test_pad_tokens_list_vector_length_1(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS.copy()

        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids)
        self.assertEqual(prompt_token_ids, expanded_prompt_token_ids)

        test_embedding_1v = torch.randn([1, 768])
        test_embedding_1v_token = "<inversion-trigger-vector-length-1>"
        test_embedding_1v_token_id = tim.add_textual_inversion(test_embedding_1v_token, test_embedding_1v)
        self.assertEqual(test_embedding_1v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids_1v_append = prompt_token_ids + [test_embedding_1v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_1v_append)
        self.assertEqual(prompt_token_ids_1v_append, expanded_prompt_token_ids)

        # at the start
        prompt_token_ids_1v_prepend = [test_embedding_1v_token_id] + prompt_token_ids
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_1v_prepend)
        self.assertEqual(prompt_token_ids_1v_prepend, expanded_prompt_token_ids)

        # in the middle
        prompt_token_ids_1v_insert = prompt_token_ids[0:2] + [test_embedding_1v_token_id] + prompt_token_ids[2:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_1v_insert)
        self.assertEqual(prompt_token_ids_1v_insert, expanded_prompt_token_ids)

    def test_pad_tokens_list_vector_length_2(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS.copy()

        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids)
        self.assertEqual(prompt_token_ids, expanded_prompt_token_ids)

        test_embedding_2v = torch.randn([2, 768])
        test_embedding_2v_token = "<inversion-trigger-vector-length-2>"
        test_embedding_2v_token_id = tim.add_textual_inversion(test_embedding_2v_token, test_embedding_2v)
        self.assertEqual(test_embedding_2v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids_2v_append = prompt_token_ids + [test_embedding_2v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_2v_append)
        self.assertNotEqual(prompt_token_ids_2v_append, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids + [test_embedding_2v_token_id, tim.clip_embedder.tokenizer.pad_token_id], expanded_prompt_token_ids)

        # at the start
        prompt_token_ids_2v_prepend = [test_embedding_2v_token_id] + prompt_token_ids
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_2v_prepend)
        self.assertNotEqual(prompt_token_ids_2v_prepend, expanded_prompt_token_ids)
        self.assertEqual([test_embedding_2v_token_id, tim.clip_embedder.tokenizer.pad_token_id] + prompt_token_ids, expanded_prompt_token_ids)

        # in the middle
        prompt_token_ids_2v_insert = prompt_token_ids[0:2] + [test_embedding_2v_token_id] + prompt_token_ids[2:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_2v_insert)
        self.assertNotEqual(prompt_token_ids_2v_insert, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids[0:2] + [test_embedding_2v_token_id, tim.clip_embedder.tokenizer.pad_token_id] + prompt_token_ids[2:3], expanded_prompt_token_ids)

    def test_pad_tokens_list_vector_length_8(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS.copy()

        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids)
        self.assertEqual(prompt_token_ids, expanded_prompt_token_ids)

        test_embedding_8v = torch.randn([8, 768])
        test_embedding_8v_token = "<inversion-trigger-vector-length-8>"
        test_embedding_8v_token_id = tim.add_textual_inversion(test_embedding_8v_token, test_embedding_8v)
        self.assertEqual(test_embedding_8v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids_8v_append = prompt_token_ids + [test_embedding_8v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_8v_append)
        self.assertNotEqual(prompt_token_ids_8v_append, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids + [test_embedding_8v_token_id] + [tim.clip_embedder.tokenizer.pad_token_id]*7, expanded_prompt_token_ids)

        # at the start
        prompt_token_ids_8v_prepend = [test_embedding_8v_token_id] + prompt_token_ids
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_8v_prepend)
        self.assertNotEqual(prompt_token_ids_8v_prepend, expanded_prompt_token_ids)
        self.assertEqual([test_embedding_8v_token_id] + [tim.clip_embedder.tokenizer.pad_token_id]*7 + prompt_token_ids, expanded_prompt_token_ids)

        # in the middle
        prompt_token_ids_8v_insert = prompt_token_ids[0:2] + [test_embedding_8v_token_id] + prompt_token_ids[2:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids=prompt_token_ids_8v_insert)
        self.assertNotEqual(prompt_token_ids_8v_insert, expanded_prompt_token_ids)
        self.assertEqual(prompt_token_ids[0:2] + [test_embedding_8v_token_id] + [tim.clip_embedder.tokenizer.pad_token_id]*7 + prompt_token_ids[2:3], expanded_prompt_token_ids)


    def test_overwrite_textual_inversion_noop(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           KNOWN_WORDS_TOKEN_IDS + \
                           (77-4) * [tim.clip_embedder.tokenizer.eos_token_id]
        prompt_embeddings = torch.randn([77, 768])

        # add embedding
        test_embedding_1v = torch.randn([1, 768])
        test_embedding_1v_token = "<inversion-trigger-vector-length-1>"
        test_embedding_1v_token_id = tim.add_textual_inversion(test_embedding_1v_token, test_embedding_1v)
        self.assertEqual(test_embedding_1v_token_id, len(KNOWN_WORDS))

        overwritten_embeddings = tim.overwrite_textual_inversion_embeddings(prompt_token_ids, prompt_embeddings)
        self.assertTrue(torch.equal(prompt_embeddings, overwritten_embeddings))

    def test_overwrite_textual_inversion_1v_single(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        default_prompt_embeddings = torch.randn([77, 768])

        # add embedding
        test_embedding_1v = torch.randn([1, 768])
        test_embedding_1v_token = "<inversion-trigger-vector-length-1>"
        test_embedding_1v_token_id = tim.add_textual_inversion(test_embedding_1v_token, test_embedding_1v)
        self.assertEqual(test_embedding_1v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS + [test_embedding_1v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:4], default_prompt_embeddings[0:4]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[4], test_embedding_1v[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[5:77], default_prompt_embeddings[5:77]))

        # at the start
        prompt_token_ids = [test_embedding_1v_token_id] + KNOWN_WORDS_TOKEN_IDS
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:1], default_prompt_embeddings[0:1]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[1], test_embedding_1v[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[2:77], default_prompt_embeddings[2:77]))

        # in the middle
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS[0:1] + [test_embedding_1v_token_id] + KNOWN_WORDS_TOKEN_IDS[1:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:2], default_prompt_embeddings[0:2]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[2], test_embedding_1v[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[3:77], default_prompt_embeddings[3:77]))



    def test_overwrite_textual_inversion_1v_multiple(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        default_prompt_embeddings = torch.randn([77, 768])

        # add embeddings
        test_embedding_1v_1 = torch.randn([1, 768])
        test_embedding_1v_1_token = "<inversion-trigger-vector-length-1-a>"
        test_embedding_1v_1_token_id = tim.add_textual_inversion(test_embedding_1v_1_token, test_embedding_1v_1)
        self.assertEqual(test_embedding_1v_1_token_id, len(KNOWN_WORDS))

        test_embedding_1v_2 = torch.randn([1, 768])
        test_embedding_1v_2_token = "<inversion-trigger-vector-length-1-b>"
        test_embedding_1v_2_token_id = tim.add_textual_inversion(test_embedding_1v_2_token, test_embedding_1v_2)
        self.assertEqual(test_embedding_1v_2_token_id, len(KNOWN_WORDS)+1)

        # at the end
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS + [test_embedding_1v_1_token_id, test_embedding_1v_2_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:4], default_prompt_embeddings[0:4]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[4], test_embedding_1v_1[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[5], test_embedding_1v_2[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[6:77], default_prompt_embeddings[6:77]))

        # at the start
        prompt_token_ids = [test_embedding_1v_1_token_id, test_embedding_1v_2_token_id] + KNOWN_WORDS_TOKEN_IDS
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:1], default_prompt_embeddings[0:1]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[1], test_embedding_1v_1[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[2], test_embedding_1v_2[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[3:77], default_prompt_embeddings[3:77]))

        # clumped in the middle
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS[0:1] + [test_embedding_1v_1_token_id, test_embedding_1v_2_token_id] + KNOWN_WORDS_TOKEN_IDS[1:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:2], default_prompt_embeddings[0:2]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[2], test_embedding_1v_1[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[3], test_embedding_1v_2[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[4:77], default_prompt_embeddings[4:77]))

        # scattered
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS[0:1] + [test_embedding_1v_1_token_id] + KNOWN_WORDS_TOKEN_IDS[1:2] + [test_embedding_1v_2_token_id] + KNOWN_WORDS_TOKEN_IDS[2:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:2], default_prompt_embeddings[0:2]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[2], test_embedding_1v_1[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[3], default_prompt_embeddings[3]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[4], test_embedding_1v_2[0]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[5:77], default_prompt_embeddings[5:77]))

    def test_overwrite_textual_inversion_4v_single(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        default_prompt_embeddings = torch.randn([77, 768])

        # add embedding
        test_embedding_4v = torch.randn([4, 768])
        test_embedding_4v_token = "<inversion-trigger-vector-length-4>"
        test_embedding_4v_token_id = tim.add_textual_inversion(test_embedding_4v_token, test_embedding_4v)
        self.assertEqual(test_embedding_4v_token_id, len(KNOWN_WORDS))

        # at the end
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS + [test_embedding_4v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:4], default_prompt_embeddings[0:4]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[4:8], test_embedding_4v))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[8:77], default_prompt_embeddings[8:77]))

        # at the start
        prompt_token_ids = [test_embedding_4v_token_id] + KNOWN_WORDS_TOKEN_IDS
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:1], default_prompt_embeddings[0:1]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[1:5], test_embedding_4v))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[5:77], default_prompt_embeddings[5:77]))

        # in the middle
        prompt_token_ids = KNOWN_WORDS_TOKEN_IDS[0:1] + [test_embedding_4v_token_id] + KNOWN_WORDS_TOKEN_IDS[1:3]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:2], default_prompt_embeddings[0:2]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[2:6], test_embedding_4v))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[6:77], default_prompt_embeddings[6:77]))

    def test_overwrite_textual_inversion_4v_overflow(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        default_prompt_embeddings = torch.randn([77, 768])

        # add embedding
        test_embedding_4v = torch.randn([4, 768])
        test_embedding_4v_token = "<inversion-trigger-vector-length-4>"
        test_embedding_4v_token_id = tim.add_textual_inversion(test_embedding_4v_token, test_embedding_4v)
        self.assertEqual(test_embedding_4v_token_id, len(KNOWN_WORDS))

        base_prompt = KNOWN_WORDS_TOKEN_IDS * 24

        # at the end
        prompt_token_ids = base_prompt + [test_embedding_4v_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        base_prompt_length = len(base_prompt)
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:base_prompt_length+1], default_prompt_embeddings[0:base_prompt_length+1]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[base_prompt_length+1:base_prompt_length+1+3], test_embedding_4v[0:3]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[base_prompt_length+1+3:77], default_prompt_embeddings[base_prompt_length+1+3:77]))

        # at the start
        prompt_token_ids = [test_embedding_4v_token_id] + base_prompt
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        expanded_prompt_token_ids = expanded_prompt_token_ids[0:75]
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:1], default_prompt_embeddings[0:1]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[1:5], test_embedding_4v))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[5:77], default_prompt_embeddings[5:77]))

        # in the middle
        prompt_token_ids = base_prompt[0:20] + [test_embedding_4v_token_id] + base_prompt[20:-1]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:21], default_prompt_embeddings[0:21]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[21:25], test_embedding_4v))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[25:77], default_prompt_embeddings[25:77]))


    def test_overwrite_textual_inversion_4v_multiple(self):
        tim = TextualInversionManager(DummyClipEmbedder())
        default_prompt_embeddings = torch.randn([77, 768])

        # add embedding
        test_embedding_4v_1 = torch.randn([4, 768])
        test_embedding_4v_1_token = "<inversion-trigger-vector-length-4-a>"
        test_embedding_4v_1_token_id = tim.add_textual_inversion(test_embedding_4v_1_token, test_embedding_4v_1)
        self.assertEqual(test_embedding_4v_1_token_id, len(KNOWN_WORDS))

        test_embedding_4v_2 = torch.randn([4, 768])
        test_embedding_4v_2_token = "<inversion-trigger-vector-length-4-b>"
        test_embedding_4v_2_token_id = tim.add_textual_inversion(test_embedding_4v_2_token, test_embedding_4v_2)
        self.assertEqual(test_embedding_4v_2_token_id, len(KNOWN_WORDS)+1)

        base_prompt = KNOWN_WORDS_TOKEN_IDS * 20

        # at the end
        prompt_token_ids = base_prompt + [test_embedding_4v_1_token_id] + [test_embedding_4v_2_token_id]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        base_prompt_length = len(base_prompt)
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:base_prompt_length+1], default_prompt_embeddings[0:base_prompt_length+1]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[base_prompt_length+1:base_prompt_length+1+4], test_embedding_4v_1))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[base_prompt_length+1+4:base_prompt_length+1+4+4], test_embedding_4v_2))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[base_prompt_length+1+4+4:77], default_prompt_embeddings[base_prompt_length+1+4+4:77]))

        # at the start
        prompt_token_ids = [test_embedding_4v_1_token_id] + [test_embedding_4v_2_token_id] + base_prompt
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        expanded_prompt_token_ids = expanded_prompt_token_ids[0:75]
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:1], default_prompt_embeddings[0:1]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[1:5], test_embedding_4v_1))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[5:9], test_embedding_4v_2))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[9:77], default_prompt_embeddings[9:77]))

        # in the middle
        prompt_token_ids = base_prompt[0:10] + [test_embedding_4v_1_token_id] + base_prompt[10:20] + [test_embedding_4v_2_token_id] + base_prompt[20:-1]
        expanded_prompt_token_ids = tim.expand_textual_inversion_token_ids(prompt_token_ids)
        padded_prompt_token_ids = [tim.clip_embedder.tokenizer.bos_token_id] + \
                           expanded_prompt_token_ids + \
                           (76 - len(expanded_prompt_token_ids)) * [tim.clip_embedder.tokenizer.eos_token_id]

        overwritten_prompt_embeddings = tim.overwrite_textual_inversion_embeddings(padded_prompt_token_ids, default_prompt_embeddings)
        self.assertFalse(torch.equal(default_prompt_embeddings, overwritten_prompt_embeddings))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[0:11], default_prompt_embeddings[0:11]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[11:15], test_embedding_4v_1))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[15:25], default_prompt_embeddings[15:25]))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[25:29], test_embedding_4v_2))
        self.assertTrue(torch.equal(overwritten_prompt_embeddings[29:77], default_prompt_embeddings[29:77]))
