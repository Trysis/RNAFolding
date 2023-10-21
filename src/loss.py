import tensorflow as tf

def masked_loss_fn(y_true, y_pred, padded_value=0.0):
    nan_mask = tf.math.is_nan(y_true)  # NaN masking
    pad_mask = tf.math.equal(y_true, padded_value)  # Padding Masking
    composite_mask = tf.math.logical_or(nan_mask, pad_mask)  # Both

    #
    y_true_modif = tf.where(composite_mask, 0.0, y_true)  # Pad & NaN = 0.0
    squared_difference = tf.square(y_true_modif - y_pred)  # MSE
    weight_mask = 1 - tf.cast(composite_mask, y_true.dtype)  # Weights

    # Only available values will be calculated
    return tf.reduce_mean(squared_difference * weight_mask, axis=-1)  # Note the `axis=-1`
