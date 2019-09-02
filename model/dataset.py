"""Create the input data pipeline using `tf.data`"""

from typing import Generator, Optional

from bert_serving.client import ConcurrentBertClient
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def _generate_data(
    filename: str, bc: ConcurrentBertClient, encoder: LabelEncoder
) -> Generator:
    with open(train_fp) as csvfile:
        csv_reader = csv.reader(csvfile)
        columns = next(csv_reader, None)

        username_idx = columns.index("username")
        text_idx = columns.index("raw_text")

        for row in csv_reader:
            vector = bc.encode([row[text_idx].strip()])
            label = encoder.transform([row[username_idx]])

            yield tf.squeeze(vector), tf.squeeze(label)


def training_dataset(
    filename: str,
    embedding_size: int,
    bc: ConcurrentBertClient,
    encoder: LabelEncoder,
    validation_size: Optional[int] = None,
):
    gen = _generate_data(filename, bc, encoder)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([embedding_size]), tf.TensorShape([1])),
    )

    # We can only batch as fast as bert server
    batch_size = bc.server_config["max_batch_size"]

    # Shuffle the examples.
    dataset = dataset.shuffle(buffer_size=10 * batch_size)

    # Repeat infinitely.
    dataset = dataset.repeat(None)

    # Create batches with `batch_size`
    dataset = dataset.batch(batch_size)

    # Prefetch to improve speed of the input pipeline.
    dataset = dataset.prefetch(buffer_size=10)

    return dataset
