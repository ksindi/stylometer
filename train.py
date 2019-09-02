#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the model

$ python train.py --data_dir ./data/
"""
import argparse
import datetime
import json
import os
import random

from absl import logging
from bert_serving.client import ConcurrentBertClient
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp

from model import params
from model import model
from model import dataset

logging.set_verbosity(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bert_port",
    type=int,
    default=5555,
    help="Port for pushing data from bert client to server",
)
parser.add_argument(
    "--bert_port_out",
    type=int,
    default=5556,
    help="Port for publishing results from bert server to client",
)
parser.add_argument(
    "--data_dir",
    default="/tmp/data",
    help="Directory containing the training and test dataset",
)
parser.add_argument(
    "--model_dir",
    default="experiments/batch_all",
    help="Experiment directory containing params.json",
)
args = parser.parse_args()

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([768]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([5]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001]))
HP_MARGIN = hp.HParam('margin', hp.RealInterval(0.5, 1.0))

data_fp = os.path.join(args.data_dir, "data.csv")
assert os.path.isfile(data_fp), f"No data file found at {data_fp}"

labels_fp = os.path.join(args.data_dir, "labels.txt")
assert os.path.isfile(labels_fp), f"No labels file found at {labels_fp}"

encoder = LabelEncoder()
with open(labels_fp) as f:
    lines = f.read().splitlines()
    encoder.fit(lines)

model = model.create_model(hparams)

log_dir = "logs/fit/" + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)
os.makedirs("training_checkpoints/", exist_ok=True)

train = dataset.training_dataset(data_fp, hparams)

history = model.fit(
    train,
    epochs=params.num_epochs,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            "training_checkpoints/weights.{epoch:02d}-{loss:.2f}.hdf5", save_freq=5
        ),
        hp.KerasCallback(logdir, hparams),  # log hparams
        tf.keras.callbacks.EarlyStopping(patience=5),
    ],
)

print(model.summary())
# tf.keras.utils.plot_model(simple_model, 'flower_model_with_shape_info.png', show_shapes=True)
