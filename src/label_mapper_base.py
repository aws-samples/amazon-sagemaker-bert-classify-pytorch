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

class LabelMapperBase:
    """
    Base class for mapping labels to zero indexed integers
    """

    def map(self, item) -> int:
        """
        Maps the raw label to corresponding zero indexed integer. E.g. if the raw labels are "Positive" & "Negative", then the corresponding integers would be 0,1
        :param item: The raw label to map. e.g. "positive"
        :return: returns the corresponding zero indexed integer, e.g. 1
        """
        raise NotImplementedError

    def reverse_map(self, item: int):
        """
        Reverse maps the integer label to corresponding raw labels. E.g. if the integer labels are 0,1, then the corresponding raw labels are "Positive" & "Negative"
        :param item: The int label to map. e.g. 1
        :return: returns the corresponding raw label , e.g. "Positive"
        """
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """
        The total number of unique classes. E.g. if you are performing sentiment analysis for positive, negative & neutral, then you would return 3
        :return: The total number of unique classes
        """
        raise NotImplementedError

    @property
    def positive_label(self):
        """
        The raw positive label. Useful for unbalanced dataset when you want to use F-score as the measure
        :return: The raw positive label , e.g. "positive"
        """
        raise NotImplementedError

    @property
    def positive_label_index(self) -> int:
        """
        The raw positive label index
        :return: The integer index corresponding to the raw positive_label
        """
        raise NotImplementedError
