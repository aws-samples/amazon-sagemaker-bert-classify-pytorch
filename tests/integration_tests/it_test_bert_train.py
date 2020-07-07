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
import sys
import tempfile
from unittest import TestCase

from builder import Builder


class ItTestBertTrain(TestCase):
    """
    Integration test
    """

    def setUp(self):
        logging.basicConfig(level="INFO", handlers=[logging.StreamHandler(sys.stdout)],
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def test_run_train(self):
        """
        Test case  run train without exception
        :return:
        """

        checkpoint_dir = None
        epochs = 20
        earlystoppingpatience = 10
        modeldir = tempfile.mkdtemp()
        batch_size = 4
        lr = 0.001
        grad_acc_steps = 2

        train_data_file = os.path.join(os.path.dirname(__file__), "..", "sample_dbpedia.csv")
        val_data_file = os.path.join(os.path.dirname(__file__), "..", "sample_dbpedia.csv")
        labels_file = os.path.join(os.path.dirname(__file__), "..", "classes.txt")
        b = Builder(train_data=train_data_file, val_data=val_data_file, labels_file=labels_file, model_dir=modeldir,
                    checkpoint_dir=checkpoint_dir, epochs=epochs,
                    early_stopping_patience=earlystoppingpatience, batch_size=batch_size,
                    grad_accumulation_steps=grad_acc_steps, learning_rate=lr)

        trainer = b.get_trainer()

        train_dataloader, val_dataloader = b.get_train_val_dataloader()

        # Act
        trainer.run_train(train_iter=train_dataloader,
                          validation_iter=val_dataloader,
                          optimizer=b.get_optimiser(),
                          model_network=b.get_network(),
                          loss_function=b.get_loss_function(), pos_label=0)
