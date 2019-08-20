#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the model"""

import argparse
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

logging.info("ConcurrentBertClient initialized")

json_path = os.path.join(args.model_dir, "params.json")
assert os.path.isfile(json_path), f"No configuration file found at {json_path}"

train_fp = os.path.join(args.data_dir, "train.csv")
eval_fp = os.path.join(args.data_dir, "eval.csv")
assert os.path.isfile(train_fp), f"No train file found at {train_fp}"
assert os.path.isfile(eval_fp), f"No eval file found at {eval_fp}"

params = Params(json_path)

model = model.StylometerModel()

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)

steps_per_epoch = round(num_train) // params.batch_size
validation_steps = round(num_val) // params.batch_size

train = dataset.train(train_fp)
validation = dataset.validation(eval_fp)

model.compile(
    optimizer="adam",
    loss=tf.contrib.losses.metric_learning.triplet_semihard_loss,
    metrics=["accuracy"],
)

print(model.summary())
# tf.keras.utils.plot_model(simple_model, 'flower_model_with_shape_info.png', show_shapes=True)

# Creating Keras callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5", period=5
)
os.makedirs("training_checkpoints/", exist_ok=True)
early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(patience=5)

history = model.fit(
    train.repeat(),
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation.repeat(),
    validation_steps=validation_steps,
    callbacks=[
        tensorboard_callback,
        model_checkpoint_callback,
        early_stopping_checkpoint,
    ],
)
