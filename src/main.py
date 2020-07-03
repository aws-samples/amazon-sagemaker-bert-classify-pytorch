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
import argparse
import logging
import os
import pickle
import sys

from builder import Builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train file wrt to train  dir", required=True)
    parser.add_argument("--traindir",
                        help="The input train  dir", default=os.environ.get("SM_CHANNEL_TRAIN", "."))

    parser.add_argument("--valfile",
                        help="The input val file wrt to val  dir", required=True)
    parser.add_argument("--valdir",
                        help="The input val dir", default=os.environ.get("SM_CHANNEL_VAL", "."))

    parser.add_argument("--classfile",
                        help="The classes txt file which is a list of classes for dbpedia", required=True)
    parser.add_argument("--classdir",
                        help="The class file  dir", default=os.environ.get("SM_CHANNEL_CLASS", "."))

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))
    parser.add_argument("--modeldir", help="The model dir", default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--checkpointdir", help="The checkpoint dir", default=None)
    parser.add_argument("--checkpointfreq",
                        help="The checkpoint frequency, only applies if the checkpoint dir is set", default=1)

    parser.add_argument("--earlystoppingpatience", help="The number of patience epochs", type=int,
                        default=10)
    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--gradaccumulation", help="The number of gradient accumulation steps", type=int, default=1)
    parser.add_argument("--batch", help="The batchsize", type=int, default=32)

    parser.add_argument("--lr", help="The learning rate", type=float, default=0.0001)
    parser.add_argument("--finetune", help="Fine tunes the final layer (classifier) model instead of the entire model",
                        type=int, default=0, choices={1, 0})
    parser.add_argument("--maxseqlen",
                        help="The max sequence len, any input that is greater than this will be truncated and fed into the network. If too large, the the bert model will not support it or you will end up Cuda OOM error! ",
                        type=int, default=256)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    train_data_file = os.path.join(args.traindir, args.trainfile)
    val_data_file = os.path.join(args.valdir, args.valfile)
    classes_file = os.path.join(args.classdir, args.classfile)

    b = Builder(train_data=train_data_file, val_data=val_data_file, labels_file=classes_file,
                checkpoint_dir=args.checkpointdir, epochs=args.epochs,
                early_stopping_patience=args.earlystoppingpatience, batch_size=args.batch, max_seq_len=args.maxseqlen,
                learning_rate=args.lr, fine_tune=args.finetune, model_dir=args.modeldir)

    trainer = b.get_trainer()

    # Persist mapper so it case be used in inference
    label_mapper_pickle_file = os.path.join(args.modeldir, "label_mapper.pkl")
    with open(label_mapper_pickle_file, "wb") as f:
        pickle.dump(b.get_label_mapper(), f)

    # Persist tokensier
    preprocessor_pickle_file = os.path.join(args.modeldir, "preprocessor.pkl")
    with open(preprocessor_pickle_file, "wb") as f:
        pickle.dump(b.get_preprocessor(), f)

    # Run training
    train_dataloader, val_dataloader = b.get_train_val_dataloader()
    trainer.run_train(train_iter=train_dataloader,
                      validation_iter=val_dataloader,
                      model_network=b.get_network(),
                      loss_function=b.get_loss_function(),
                      optimizer=b.get_optimiser(), pos_label=b.get_pos_label_index())


if "__main__" == __name__:
    main()
