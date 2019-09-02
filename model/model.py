import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp


def create_model(embedding_size: int, hparams: hp.HParam):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(embedding_size, activation=None),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hparams["learning_rate"]),
        loss=tfa.losses.TripletSemiHardLoss(hparams["margin"]),
    )

    return model
