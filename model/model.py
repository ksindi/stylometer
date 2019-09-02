import tensorflow as tf
import tensorflow_addons as tfa

def create_model(hparams):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(hparams['num_units'], activation=None),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hparams['learning_rate']),
        loss=tfa.losses.TripletSemiHardLoss(hparams['margin']),
    )

    return model
