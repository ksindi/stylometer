#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the model"""

import argparse
import datetime
import json
import os
import random

from absl import logging
import tensorflow as tf
from bert_serving.client import ConcurrentBertClient

from model import params
from model import model
from model import dataset

logging.set_verbosity(logging.INFO)

tf.executing_eagerly()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    default="experiments/batch_all",
    help="Experiment directory containing params.json",
)
parser.add_argument(
    "--data_dir",
    default="/tmp/data",
    help="Directory containing the training and test dataset",
)
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
args = parser.parse_args()
print("Args: ", args)

bc = ConcurrentBertClient(port=args.bert_port, port_out=args.bert_port_out)

logging.info("BertClient initialized")

json_path = os.path.join(args.model_dir, "params.json")
assert os.path.isfile(json_path), f"No configuration file found at {json_path}"

train_fp = os.path.join(args.data_dir, "train.tfrecord")
assert os.path.isfile(train_fp), f"No train file found at {train_fp}"
eval_fp = os.path.join(args.data_dir, "eval.tfrecord")
assert os.path.isfile(eval_fp), f"No validation file found at {eval_fp}"

params = params.Params(json_path)

model = model.StylometerModel(params)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)

train = dataset.training_dataset(train_fp, params)
validation = dataset.training_dataset(eval_fp, params)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  # tf.contrib.losses.metric_learning.triplet_semihard_loss,
    metrics=["accuracy"],  # keras_metrics.precision(), keras_metrics.recall()
)

# Creating Keras callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "training_checkpoints/weights.{epoch:02d}.hdf5",  # -{val_loss:.2f}
    save_freq=5,
    monitor="val_loss",
)
os.makedirs("training_checkpoints/", exist_ok=True)
early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(patience=5)

history = model.fit(
    train,
    # epochs=5,
    # batch_size=params.batch_size,
    shuffle=True,
    steps_per_epoch=100,
    validation_data=validation,
    validation_steps=10,
    callbacks=[
        tensorboard_callback,
        model_checkpoint_callback,
        # early_stopping_checkpoint,
    ],
)

print(model.summary())
# tf.keras.utils.plot_model(simple_model, 'flower_model_with_shape_info.png', show_shapes=True)
