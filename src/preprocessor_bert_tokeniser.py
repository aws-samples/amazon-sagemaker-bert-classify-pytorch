# ***************************************************************************************
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.                    *
#                                                                                       *
# Permission is hereby granted, free of charge, to any person obtaining a copy of this  *
# software and associated documentation files (the "Software"), to deal in the Software *
# without restriction, including without limitation the rights to use, copy, modify,    *
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to    *
# permit persons to whom the Software is furnished to do so.                            *
#                                                                                       *
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,   *
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A         *
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT    *
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION     *
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE        *
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                *
# ***************************************************************************************
import torch


class PreprocessorBertTokeniser:
    """
    Text to an array of indices using the BERT tokeniser
    """

    def __init__(self, max_feature_len, tokeniser):
        self.max_feature_len = max_feature_len
        self.tokeniser = tokeniser
        self.item = None

    @staticmethod
    def pad_token():
        return "[PAD]"

    @staticmethod
    def eos_token():
        return "<EOS>"

    @staticmethod
    def unk_token():
        return "[UNK]"

    def __call__(self, item):
        self.item = item
        self.tokenise() \
            .sequence_pad() \
            .token_to_index() \
            .to_tensor()

        return self.item

    def tokenise(self):
        """
        Converts text to tokens, e.g. "The dog" would return ["The", "dog"]
        """
        tokens = self.tokeniser.tokenize(self.item)
        self.item = tokens
        return self

    def token_to_index(self):
        """
        Converts a string of token to corresponding indices. e.g. ["The", "dog"] would return [2,3]
        :return: self
        """
        result = self.tokeniser.convert_tokens_to_ids(self.item)
        self.item = result
        return self

    def sequence_pad(self):
        """
        Converts the tokens to fixed size and formats it according to bert
        :return: self
        """
        tokens = self.item[:self.max_feature_len - 2]
        pad_tokens = [self.pad_token()] * (self.max_feature_len - 2 - len(tokens))
        result = ['[CLS]'] + tokens + pad_tokens + ['[SEP]']

        self.item = result
        return self

    def to_tensor(self):
        """
        Converts list of int to tensor
        :return: self
        """

        self.item = torch.tensor(self.item)
        return self
