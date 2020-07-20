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
import datetime
import glob
import logging
import os

import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score


class Train:
    """
    Trains on GPU / CPU
    """

    def __init__(self, model_dir, device=None, epochs=10, early_stopping_patience=20, checkpoint_frequency=1,
                 checkpoint_dir=None,
                 accumulation_steps=1):
        self.model_dir = model_dir
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.snapshotter = None

        # Set up device is not set
        available_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            available_device = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

        self.device = device or available_device

        # Assume multi gpu if device passed is a list and not a string
        self._is_multigpu = not isinstance(self.device, str)

        self._default_device = self.device[0] if self._is_multigpu else self.device

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def snapshot(self, model, model_dir, prefix="best_snaphsot"):
        snapshot_prefix = os.path.join(model_dir, prefix)
        snapshot_path = snapshot_prefix + 'model.pt'

        self._logger.info("Snapshot model to {}".format(snapshot_path))

        # If nn.dataparallel, get the underlying module
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        torch.save(model, snapshot_path)

    def run_train(self, train_iter, validation_iter, model_network, loss_function, optimizer, pos_label):
        """
    Runs train...
        :param pos_label:
        :param validation_iter: Validation set
        :param train_iter: Train Data
        :param model_network: A neural network
        :param loss_function: Pytorch loss function
        :param optimizer: Optimiser
        """
        best_results = None
        start = datetime.datetime.now()
        iterations = 0
        val_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
        val_log_template = "Run {}".format(val_log_template)

        best_score = None

        no_improvement_epochs = 0

        if self._is_multigpu:
            model_network = nn.DataParallel(model_network, device_ids=self.device, output_device=self._default_device)
            self._logger.info("Using multi gpu with devices {}, default {} ".format(self.device, self._default_device))

        model_network.to(device=self._default_device)

        for epoch in range(self.epochs):
            losses_train = []
            actual_train = []
            predicted_train = []

            self._logger.debug("Running epoch {}".format(self.epochs))

            for idx, batch in enumerate(train_iter):
                self._logger.debug("Running batch {}".format(idx))
                batch_x = batch[0].to(device=self._default_device)
                batch_y = batch[1].to(device=self._default_device)

                self._logger.debug("batch x shape is {}".format(batch_x.shape))

                iterations += 1

                # Step 2. train
                model_network.train()
                model_network.zero_grad()

                # Step 3. Run the forward pass
                # words
                self._logger.debug("Running forward")
                predicted = model_network(batch_x)[0]

                # Step 4. Compute loss
                self._logger.debug("Running loss")
                loss = loss_function(predicted, batch_y) / self.accumulation_steps
                loss.backward()

                losses_train.append(loss.item())
                actual_train.extend(batch_y.cpu().tolist())
                predicted_train.extend(torch.max(predicted, 1)[1].view(-1).cpu().tolist())

                # Step 5. Only update weights after gradients are accumulated for n steps
                if (idx + 1) % self.accumulation_steps == 0:
                    self._logger.debug("Running optimiser")
                    optimizer.step()
                    model_network.zero_grad()

            # Print training set results
            self._logger.info("Train set result details:")
            train_loss = sum(losses_train) / len(losses_train)
            train_score = accuracy_score(actual_train, predicted_train)
            self._logger.info("Train set result details: {}".format(train_score))

            # Print validation set results
            self._logger.info("Validation set result details:")
            val_actuals, val_predicted, val_loss = self.validate(loss_function, model_network, validation_iter)
            val_score = accuracy_score(val_actuals, val_predicted)
            self._logger.info("Validation set result details: {} ".format(val_score))

            # Snapshot best score
            if best_score is None or val_score > best_score:
                best_results = (val_score, val_actuals, val_predicted)
                self._logger.info(
                    "Snapshotting because the current score {} is greater than {} ".format(val_score, best_score))
                self.snapshot(model_network, model_dir=self.model_dir)
                best_score = val_score
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # Checkpoint
            if self.checkpoint_dir and (epoch % self.checkpoint_frequency == 0):
                self.create_checkpoint(model_network, self.checkpoint_dir)

            # evaluate performance on validation set periodically
            self._logger.info(val_log_template.format((datetime.datetime.now() - start).seconds,
                                                      epoch, iterations, 1 + len(batch_x), len(train_iter),
                                                      100. * (1 + len(batch_x)) / len(train_iter), train_loss,
                                                      val_loss, train_score,
                                                      val_score))

            print("###score: train_loss### {}".format(train_loss))
            print("###score: val_loss### {}".format(val_loss))
            print("###score: train_score### {}".format(train_score))
            print("###score: val_score### {}".format(val_score))

            if no_improvement_epochs > self.early_stopping_patience:
                self._logger.info("Early stopping.. with no improvement in {}".format(no_improvement_epochs))
                break

        return best_results

    def validate(self, loss_function, model_network, val_iter):
        # switch model to evaluation mode
        model_network.eval()

        # total loss
        val_loss = 0

        actuals = torch.tensor([], dtype=torch.long).to(device=self._default_device)
        predicted = torch.tensor([], dtype=torch.long).to(device=self._default_device)

        with torch.no_grad():
            for idx, val in enumerate(val_iter):
                val_batch_idx = val[0].to(device=self._default_device)
                val_y = val[1].to(device=self._default_device)

                pred_batch_y = model_network(val_batch_idx)[0]

                # compute loss
                val_loss += loss_function(pred_batch_y, val_y).item()

                actuals = torch.cat([actuals, val_y])
                pred_flat = torch.max(pred_batch_y, dim=1)[1].view(-1)
                predicted = torch.cat([predicted, pred_flat])

        # Average loss
        val_loss = val_loss / len(actuals)
        return actuals.cpu().tolist(), predicted.cpu().tolist(), val_loss

    def create_checkpoint(self, model, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

        self._logger.info("Checkpoint model to {}".format(checkpoint_path))

        # If nn.dataparallel, get the underlying module
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        torch.save({
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)

    def try_load_model_from_checkpoint(self):
        loaded_weights = None
        if self.checkpoint_dir is not None:
            model_files = list(glob.glob("{}/*.pt".format(self.checkpoint_dir)))
            if len(model_files) > 0:
                model_file = model_files[0]
                self._logger.info(
                    "Loading checkpoint {} , found {} checkpoint files".format(model_file, len(model_files)))
                checkpoint = torch.load(model_file)
                loaded_weights = checkpoint['model_state_dict']
        return loaded_weights
