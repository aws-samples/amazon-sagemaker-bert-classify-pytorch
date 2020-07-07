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
