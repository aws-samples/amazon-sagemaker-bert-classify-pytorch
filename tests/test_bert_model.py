from unittest import TestCase
import torch

import transformers

from bert_model import BertModel


class TestBertModel(TestCase):

    def test_forward(self):
        # Bert Config
        vocab_size = 10
        sequence_len = 20
        batch = 32
        num_classes = 3

        expected_shape = (batch, num_classes)

        input_batch = torch.randint(low=0, high=vocab_size - 1, size=(batch, sequence_len))
        config = transformers.BertConfig(vocab_size=vocab_size, hidden_size=10, num_hidden_layers=1,
                                         num_attention_heads=1, num_labels=num_classes)
        sut = BertModel(None, None, bert_config=config)

        # Act
        actual = sut.forward(input_batch)[0]

        # Assert
        self.assertEqual(expected_shape, actual.shape)
