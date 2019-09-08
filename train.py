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
from sklearn.preprocessing import LabelEncoder
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import tensorflow_addons as tfa


logging.set_verbosity(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    default="/tmp/data",
    help="Directory containing the training and test dataset",
)
args = parser.parse_args()

data_fp = os.path.join(args.data_dir, "train.tfrecord")
assert os.path.isfile(data_fp), f"No data file found at {data_fp}"

labels_fp = os.path.join(args.data_dir, "labels.txt")
assert os.path.isfile(labels_fp), f"No labels file found at {labels_fp}"

log_dir = "logs/fit/" + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)
os.makedirs("training_checkpoints/", exist_ok=True)

validation_size = 500
num_hidden_unit = 768
buffer_size = 100
batch_size = 64
num_epochs = 5


def _decode_record(record, num_hidden_unit: int):
    """Decodes a record to a TensorFlow example."""
    decoded = tf.io.parse_single_example(
        record,
        {
            "features": tf.io.FixedLenFeature([num_hidden_unit], tf.float32),
            "labels": tf.io.FixedLenFeature([1], tf.int64),
        },
    )
    return decoded["features"], decoded["labels"]


train = (
    tf.data.TFRecordDataset(data_fp)
    .skip(validation_size)
    .repeat()
    .shuffle(buffer_size=buffer_size)
    .map(lambda record: _decode_record(record, num_hidden_unit))
    .batch(batch_size)
    .prefetch(1)
)

validation = (
    tf.data.TFRecordDataset(data_fp)
    .take(validation_size)
    .repeat()
    .shuffle(buffer_size=buffer_size)
    .map(lambda record: _decode_record(record, num_hidden_unit))
    .batch(batch_size)
    .prefetch(1)
)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(768, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # hparams: hp.HParam
    loss=tfa.losses.TripletSemiHardLoss(),  # hparams["margin"]
)

history = model.fit(
    train,
    steps_per_epoch=100,
    epochs=num_epochs,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            "training_checkpoints/weights.{epoch:02d}-{loss:.2f}.hdf5", save_freq=5
        ),
        tf.keras.callbacks.EarlyStopping(patience=5),
        # hp.KerasCallback(logdir, hparams),  # log hparams
    ],
)

print(model.summary())
# tf.keras.utils.plot_model(simple_model, 'stylometer.png', show_shapes=True)
