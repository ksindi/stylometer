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
import tensorflow as tf
import tensorflow_addons as tfa
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
args = parser.parse_args()
print("Args: ", args)

json_path = os.path.join(args.model_dir, "params.json")
assert os.path.isfile(json_path), f"No configuration file found at {json_path}"

train_fp = os.path.join(args.data_dir, "train.tfrecord")
assert os.path.isfile(train_fp), f"No train file found at {train_fp}"
eval_fp = os.path.join(args.data_dir, "eval.tfrecord")
assert os.path.isfile(eval_fp), f"No validation file found at {eval_fp}"

params = params.Params(json_path)

# model = model.StylometerModel(params)

log_dir = "logs/fit/" + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir)

train = dataset.training_dataset(train_fp, params)
validation = dataset.training_dataset(eval_fp, params)

model = tf.keras.models.Sequential(
    [tf.keras.layers.Input(shape=(768,)), tf.keras.layers.Dense(10)]
)
model.compile(
    optimizer="adam",
    loss=tfa.losses.triplet_semihard_loss,
    metrics=["accuracy"],  # keras_metrics.precision(), keras_metrics.recall()
)

# Creating Keras callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "training_checkpoints/weights.{epoch:02d}-{loss:.2f}.hdf5",
    save_freq=5,
    monitor="val_loss",
)
os.makedirs("training_checkpoints/", exist_ok=True)
early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(patience=5)

history = model.fit(
    train,
    epochs=params.num_epochs,
    steps_per_epoch=1000,
    validation_data=validation,
    validation_steps=10,
    callbacks=[
        tensorboard_callback,
        model_checkpoint_callback,
        #early_stopping_checkpoint,
    ],
)

print(model.summary())
# tf.keras.utils.plot_model(simple_model, 'flower_model_with_shape_info.png', show_shapes=True)
