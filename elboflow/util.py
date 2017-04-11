import functools as ft
import numpy as np
import tensorflow as tf


LOG2 = np.log(2).astype(np.float32)
LOGPI = np.log(np.pi).astype(np.float32)
LOG2PI = LOG2 + LOGPI
LOG2PIE = LOG2PI + 1.0
FLOATX = tf.float32


class BaseDistribution:
    """Base class for distributions."""
    pass


def as_tensor(x):
    """Convert `x` to a tensor or distribution."""
    if isinstance(x, BaseDistribution):
        return x
    return tf.convert_to_tensor(x, FLOATX)


def assert_constant(x):
    """Assert that `x` is not a distribution."""
    assert not isinstance(x, BaseDistribution), "expected a constant but got %s" % x


@ft.wraps(np.tril)
def tril(x, k=0, name=None):
    return tf.matrix_band_part(x, -1, k, name)


get_variable = tf.get_variable


def get_positive_variable(name, shape=None, dtype=None, initializer=None, regularizer=None,
                          trainable=True, collections=None, caching_device=None, partitioner=None,
                          validate_shape=True, custom_getter=None, transform=None):
    """
    Get an existing positive variable or create a new one.
    """
    x = get_variable(name, shape, dtype, initializer, regularizer, trainable, collections,
                     caching_device, partitioner, validate_shape, custom_getter)
    transform = transform or tf.nn.softplus
    return transform(x)


def get_tril_variable(name, shape=None, dtype=None, initializer=None, regularizer=None,
                      trainable=True, collections=None, caching_device=None, partitioner=None,
                      validate_shape=True, custom_getter=None):
    """
    Get an existing lower triangular matrix variable or create a new one.
    """
    x = get_variable(name, shape, dtype, initializer, regularizer, trainable, collections,
                     caching_device, partitioner, validate_shape, custom_getter)
    return tril(x)


def get_cholesky_variable(name, shape=None, dtype=None, initializer=None, regularizer=None,
                          trainable=True, collections=None, caching_device=None, partitioner=None,
                          validate_shape=True, custom_getter=None, transform=None):
    """
    Get an existing Cholesky variable or create a new one.
    """
    x = get_tril_variable(name, shape, dtype, initializer, regularizer, trainable, collections,
                          caching_device, partitioner, validate_shape, custom_getter)
    transform = transform or tf.nn.softplus
    return tf.matrix_set_diag(x, transform(tf.matrix_diag_part(x)))


def get_positive_definite_variable(name, shape=None, dtype=None, initializer=None, regularizer=None,
                                   trainable=True, collections=None, caching_device=None,
                                   partitioner=None, validate_shape=True, custom_getter=None,
                                   transform=None):
    """
    Get an existing positive definite variable or create a new one.
    """
    x = get_cholesky_variable(name, shape, dtype, initializer, regularizer, trainable, collections,
                              caching_device, partitioner, validate_shape, custom_getter, transform)
    return tf.matmul(x, x, True)


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


def minmax(x, axis=None):
    """
    Evaluate the minimum and maximum of an array.
    """
    return np.min(x, axis), np.max(x, axis)


def add_bias(x):
    """
    Add a bias feature to a design matrix.
    """
    return np.hstack([np.ones((x.shape[0], 1)), x])
