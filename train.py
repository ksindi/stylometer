#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the model"""

import argparse
import json
import os
import random

import tensorflow as tf
from bert_serving.client import ConcurrentBertClient

from model.params import Params
from model.input_fn import train_input_fn, eval_input_fn
from model.model_fn import model_fn

tf.logging.set_verbosity(tf.logging.INFO)

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

tf.logging.info("Initialized ")

json_path = os.path.join(args.model_dir, "params.json")
assert os.path.isfile(json_path), f"No configuration file found at {json_path}"

train_fp = os.path.join(args.data_dir, "train.csv")
eval_fp = os.path.join(args.data_dir, "eval.csv")
assert os.path.isfile(train_fp), f"No train file found at {train_fp}"
assert os.path.isfile(eval_fp), f"No eval file found at {eval_fp}"

params = Params(json_path)

config = tf.estimator.RunConfig(
    tf_random_seed=42,
    model_dir=args.model_dir,
    save_summary_steps=params.save_summary_steps,
    save_checkpoints_secs=120,
)

estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_fp, params, bc))
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: eval_input_fn(eval_fp, params, bc), throttle_secs=0
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
