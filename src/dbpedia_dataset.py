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
import csv
import logging

from torch.utils.data import Dataset


class DbpediaDataset(Dataset):
    """
    Dbpedia  dataset
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __init__(self, file: str, preprocessor=None):
        self.preprocessor = preprocessor
        self._file = file

        with open(file) as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')

            self._items = [(r[2], int(r[0])) for r in reader]

        self.logger.info("Loaded {} records from the dataset".format(len(self)))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        x, y = self._items[idx]

        # The original label is 1 indexed, this needs to be converted to zero index
        y = y - 1

        if self.preprocessor:
            x = self.preprocessor(x)

        return x, y
