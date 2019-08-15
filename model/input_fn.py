"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

from bert_serving.client import ConcurrentBertClient

from model.params import Params


def train_input_fn(filenames: str, params: Params, bc: ConcurrentBertClient):
    # datatest must be tuples of the text and the label
    return (
        tf.data.experimental.CsvDataset(
            filenames=filenames,
            record_defaults=[tf.string, tf.string],
            header=True,
            field_delim=",",
            select_cols=[1, 2],  # Only parse last two columns
        )
        .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=params.buffer_size))
        .batch(params.batch_size)
        .map(
            lambda text_, label: (
                tf.py_function(bc.encode, [text_], [tf.float32], name="bert_client"),
                label,
            ),
            num_parallel_calls=params.num_parallel_calls,
        )
        #.map(lambda x, y: ({"feature": x}, y))
        .prefetch(1)  # make sure you always have one batch ready to serve
    )


def eval_input_fn(filenames: str, params: Params, bc: ConcurrentBertClient):
    # datatest must be tuples of the text and the label
    return (
        tf.data.experimental.CsvDataset(
            filenames=eval_fp,
            record_defaults=[tf.string, tf.string],
            header=True,
            field_delim=",",
            select_cols=[1, 2],  # Only parse last two columns
        )
        .batch(params.batch_size)
        .map(
            lambda text_, label: (
                tf.py_function(bc.encode, [text_], [tf.float32], name="bert_client"),
                label,
            ),
            num_parallel_calls=params.num_parallel_calls,
        )
        #.map(lambda x, y: ({"feature": x}, y))
        .prefetch(1)  # make sure you always have one batch ready to serve
    )
