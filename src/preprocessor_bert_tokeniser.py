# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import torch


class PreprocessorBertTokeniser:
    """
    Text to an array of indices using the BERT tokeniser
    """

    def __init__(self, max_feature_len, tokeniser, case_sensitive=True):
        self.max_feature_len = max_feature_len
        self.case_insensitive = not case_sensitive
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
