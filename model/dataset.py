"""Create the input data pipeline using `tf.data`"""

import typing

from bert_serving.client import ConcurrentBertClient
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def generate_data(filename: str, bc: ConcurrentBertClient, encoder: LabelEncoder) -> typing.Generator[box.Box, None, None]:
    with open(train_fp) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        colums = next(csv_reader, None)  # skip the headers

        username_idx = columns.index("username")
        text_idx = columns.index("raw_text")

        for row in csv_reader:
            vector = bc.encode([row[text_idx].strip()])
            label = encoder.transform([row[username_idx]])

            yield tf.squeeze(vector), tf.squeeze(label)


def training_dataset(g: str, hparams: Params):
    # datatest must be tuples of the text and the label
    return (
        tf.data.Dataset.from_generator(g)
        .repeat()
        .shuffle(buffer_size=params.buffer_size)
        .map(lambda record: _decode_record(record, params.num_hidden_unit))
        .batch(params.batch_size)
        .prefetch(1)
    )
