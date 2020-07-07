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
