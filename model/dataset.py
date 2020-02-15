"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

from bert_serving.client import ConcurrentBertClient

from model.params import Params


def _decode_record(record, num_hidden_unit: int, num_classes: int):
    """Decodes a record to a TensorFlow example."""
    decoded = tf.io.parse_single_example(
        record,
        {
            "features": tf.io.FixedLenFeature([num_hidden_unit], tf.float32),
            "labels": tf.io.FixedLenFeature([num_classes], tf.int64),
        },
    )
    return decoded["features"], decoded["labels"]


def training_dataset(filename: str, params: Params):
    # datatest must be tuples of the text and the label
    return (
        tf.data.TFRecordDataset(filename)
        .repeat()
        .shuffle(buffer_size=params.buffer_size)
        .map(lambda record: _decode_record(record, params.num_hidden_unit, params.num_classes))
        .batch(params.batch_size)
        .prefetch(1)
    )
