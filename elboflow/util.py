import numpy as np
import tensorflow as tf


LOG2 = np.log(2).astype(np.float32)
LOGPI = np.log(np.pi).astype(np.float32)
LOG2PI = LOG2 + LOGPI
LOG2PIE = LOG2PI + 1.0
FLOATX = tf.float32


def multidigamma(x, p):
    """
    Compute the multivariate digamma function recursively.

    References
    ----------
    https://en.wikipedia.org/wiki/Multivariate_gamma_function#Derivatives
    """
    x = as_tensor(x)
    return tf.reduce_sum(tf.digamma(x[..., None] - 0.5 * tf.range(p, dtype=x.dtype)), axis=-1)


def lmultigamma(x, p):
    """
    Compute the natural logarithm of the multivariate gamma function recursively.

    References
    ----------
    https://en.wikipedia.org/wiki/Multivariate_gamma_function
    """
    x = as_tensor(x)
    return 0.25 * p * (p - 1.0) * LOGPI + \
        tf.reduce_sum(tf.lgamma(x[..., None] - 0.5 * tf.range(p, dtype=x.dtype)), axis=-1)


def as_tensor(x):
    return tf.convert_to_tensor(x, FLOATX)
