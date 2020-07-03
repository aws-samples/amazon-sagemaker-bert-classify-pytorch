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
from unittest import TestCase

from src.preprocessor_bert_tokeniser import PreprocessorBertTokeniser


class TestPreprocessorBertTokeniser(TestCase):

    def test_sequence_short(self):
        """
        Test case  sequences that are too short should be padded
        :return:
        """
        sut = PreprocessorBertTokeniser(max_feature_len=5, tokeniser=None)
        sut.item = ["THE"]
        expected = ["[CLS]", "THE", "[PAD]", "[PAD]", "[SEP]"]

        # Act
        sut.sequence_pad()

        # Assert
        self.assertSequenceEqual(expected, sut.item)

    def test_sequence_long(self):
        """
        Test case sequences that are too long should be truncated
        :return:
        """
        sut = PreprocessorBertTokeniser(max_feature_len=5, tokeniser=None)
        sut.item = ["THE", "dog", "ate", "a", "biscuit"]
        expected = ["[CLS]", "THE", "dog", "ate", "[SEP]"]

        # Act
        sut.sequence_pad()

        # Assert
        self.assertSequenceEqual(expected, sut.item)
