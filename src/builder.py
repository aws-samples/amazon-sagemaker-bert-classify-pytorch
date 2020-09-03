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

import logging
import os

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from bert_model import BertModel
from bert_train import Train
from dbpedia_dataset import DbpediaDataset
from dbpedia_dataset_label_mapper import DbpediaLabelMapper
from preprocessor_bert_tokeniser import PreprocessorBertTokeniser


class Builder:

    def __init__(self, train_data, val_data, labels_file, model_dir, num_workers=None, checkpoint_dir=None, epochs=10,
                 early_stopping_patience=10, checkpoint_frequency=1, grad_accumulation_steps=8, batch_size=8,
                 max_seq_len=512, learning_rate=0.00001, fine_tune=True):
        self.model_dir = model_dir
        self.fine_tune = fine_tune
        self.learning_rate = learning_rate
        self.checkpoint_frequency = checkpoint_frequency
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.train_data = train_data
        self.val_data = val_data
        self.labels_file = labels_file
        self.batch_size = batch_size
        # Note: Since the max seq len for pos embedding is 512 , in the pretrained  bert this must be less than eq to 512
        # Also note increasing the length greater also will create GPU out of mememory error
        self._max_seq_len = max_seq_len
        self.num_workers = num_workers or os.cpu_count() - 1
        if self.num_workers <= 0:
            self.num_workers = 0

        self._network = None
        self._train_dataloader = None
        self._train_dataset = None
        self._val_dataset = None
        self._val_dataloader = None
        self._trainer = None
        self._lossfunc = None
        self._optimiser = None
        self._label_mapper = None

        self._bert_model_name = "bert-base-cased"
        self._token_lower_case = False

    def get_preprocessor(self):
        tokeniser = BertTokenizer.from_pretrained(self._bert_model_name, do_lower_case=self._token_lower_case)
        preprocessor = PreprocessorBertTokeniser(max_feature_len=self._max_seq_len, tokeniser=tokeniser)
        return preprocessor

    def get_network(self):
        # If network already loaded simply return
        if self._network is not None: return self._network

        # If checkpoint file is available, load from checkpoint
        self._network = self.get_trainer().try_load_model_from_checkpoint()

        # Only load from BERT pretrained when no checkpoint is available
        if self._network is None:
            self._logger.info(
                "No checkpoint models found.. Loading pretrained BERT {}".format(self._bert_model_name))
            self._network = BertModel(self._bert_model_name, self.get_label_mapper().num_classes,
                                      fine_tune=self.fine_tune)

        return self._network

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = DbpediaDataset(self.train_data, preprocessor=self.get_preprocessor())

        return self._train_dataset

    def get_val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = DbpediaDataset(self.val_data, preprocessor=self.get_preprocessor())

        return self._val_dataset

    def get_label_mapper(self):
        if self._label_mapper is None:
            self._label_mapper = DbpediaLabelMapper(self.labels_file)

        return self._label_mapper

    def get_pos_label_index(self):
        return self.get_label_mapper().positive_label_index

    def get_train_val_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(dataset=self.get_train_dataset(), num_workers=self.num_workers,
                                                batch_size=self.batch_size, shuffle=True)

        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(dataset=self.get_val_dataset(), num_workers=self.num_workers,
                                              batch_size=self.batch_size, shuffle=False)

        return self._train_dataloader, self._val_dataloader

    def get_loss_function(self):
        if self._lossfunc is None:
            self._lossfunc = nn.CrossEntropyLoss()
        return self._lossfunc

    def get_optimiser(self):
        if self._optimiser is None:
            self._optimiser = Adam(params=self.get_network().parameters(), lr=self.learning_rate)
        return self._optimiser

    def get_trainer(self):
        if self._trainer is None:
            self._trainer = Train(model_dir=self.model_dir, epochs=self.epochs,
                                  early_stopping_patience=self.early_stopping_patience,
                                  checkpoint_frequency=self.checkpoint_frequency,
                                  checkpoint_dir=self.checkpoint_dir,
                                  accumulation_steps=self.grad_accumulation_steps)

        return self._trainer

    @property
    def _logger(self):
        return logging.getLogger(__name__)
