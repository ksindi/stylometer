#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrtie training data to tfrecords
$ python tfrecords.py --data_dir ./data/
"""

import argparse
import csv
import os

from absl import logging
from bert_serving.client import ConcurrentBertClient
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tqdm

logging.set_verbosity(logging.INFO)

parser = argparse.ArgumentParser()
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

labels_fp = os.path.join(args.data_dir, "labels.txt")
assert os.path.isfile(labels_fp), f"No label file found at {labels_fp}"

train_fp = os.path.join(args.data_dir, "train.csv")
assert os.path.isfile(train_fp), f"No train file found at {train_fp}"

writer_fp = os.path.join(args.data_dir, "train.tfrecord")
# remove file if exists
try:
    os.remove(writer_fp)
except OSError:
    pass

# Create a label encoder to encode the usernames as integers
encoder = LabelEncoder()
with open(labels_fp) as f:
    lines = f.read().splitlines()
    encoder.fit(lines)

# write to tfrecord
with tf.io.TFRecordWriter(writer_fp) as writer, tqdm.tqdm() as pbar:

    def create_float_feature(values):  # numpy.ndarray
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def create_int_feature(values):  # list
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    with open(train_fp) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        next(csv_reader, None)  # skip the headers

        for row in csv_reader:
            vector = bc.encode([row[1].strip()])
            label = encoder.transform([row[2]])

            # TODO: tf.squeeze
            features = {
                "features": create_float_feature(np.squeeze(vector)),
                "labels": create_int_feature(label),
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            pbar.update(1)
