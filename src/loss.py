import tensorflow as tf

def masked_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mask = 1 - tf.cast(tf.math.is_nan(y_true), y_true.dtype)
    return tf.reduce_mean(squared_difference * mask, axis=-1)  # Note the `axis=-1`

