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
