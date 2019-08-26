#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the model"""

import argparse
import os

import tensorflow as tf

from model.input_fn import test_input_fn
from model.model_fn import model_fn
from model.params import Params


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    default="experiments/base_model",
    help="Experiment directory containing params.json",
)
parser.add_argument(
    "--data_dir", default="data/mnist", help="Directory containing the dataset"
)


if __name__ == "__main__":
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No configuration file found at {}".format(
        json_path
    )
    params = Params(json_path)

    # TODO
