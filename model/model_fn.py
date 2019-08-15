"""Define the model."""

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of embeddings
        labels: labels of the embeddings
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """

    # MODEL: define the layers of the model
    #with tf.variable_scope("model"):
    #        embeddings = tf.layers.dense(features, params.embedding_size)
    embeddings = features

    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"embeddings": embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(
            labels, embeddings, margin=params.margin, squared=params.squared
        )
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(
            labels, embeddings, margin=params.margin, squared=params.squared
        )
    else:
        raise ValueError(f"Triplet strategy not recognized: {params.triplet_strategy}")

    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all":
            eval_metric_ops["fraction_positive_triplets"] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops
        )

    # Summaries for training
    tf.summary.scalar("loss", loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar("fraction_positive_triplets", fraction)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
