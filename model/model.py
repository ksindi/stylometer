import tensorflow as tf

from model.params import Params


class StylometerModel(tf.keras.Model):
    """
    Model class.
    """

    def __init__(self, params: Params):  # triplet_strategy: str,
        """
        Args:
            triplet_strategy: Either `batch_all` or `batch_hard`.
        """
        super().__init__()
        num_classes = params.num_classes
        self.margin = params.margin
        self.squared = params.squared

        self.dense_1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def call(self, input_data):
        """
        Invoke the model.
        Args:
            embeddings: 2D Tensor, a batch of sequences of BERT embeddings.
        Returns:
            logits: Tensor [B, (1 + n), 2] containing unscaled predictions.
        """
        # [batch_size, sentence_embeddings]
        x = self.dense_1(input_data)
        x = self.dense_2(x)

        # loss, fraction = batch_all_triplet_loss(
        #    labels, embeddings, margin=self.margin, squared=self.squared
        # )

        return x


def predict_text(text, bc, maxlen, model):
    enc = preproc_text_bert(text, bc, maxlen)
    score = model.predict(enc)
    return score
