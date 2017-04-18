import functools as ft
import sys
import io
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


@ft.wraps(tf.convert_to_tensor)
def as_tensor(x, dtype=None, name=None):
    if isinstance(x, BaseDistribution):
        return x
    return tf.convert_to_tensor(x, dtype or FLOATX, name)


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


def get_normalized_variable(name, shape=None, dtype=None, initializer=None, regularizer=None,
                            trainable=True, collections=None, caching_device=None,
                            partitioner=None, validate_shape=True, custom_getter=None,
                            transform=None):
    """
    Get an existing variable or create a new one such that it sums to one along the last dimension.
    """
    x = get_variable(name, shape, dtype, initializer, regularizer, trainable, collections,
                     caching_device, partitioner, validate_shape, custom_getter)
    transform = transform or tf.nn.softmax
    return transform(x)


def multidigamma(x, p, name=None):
    """
    Compute the multivariate digamma function recursively.

    References
    ----------
    https://en.wikipedia.org/wiki/Multivariate_gamma_function#Derivatives
    """
    x = as_tensor(x)
    return tf.reduce_sum(tf.digamma(x[..., None] - 0.5 * tf.range(p, dtype=x.dtype)), axis=-1,
                         name=name)


def lmultigamma(x, p, name=None):
    """
    Compute the natural logarithm of the multivariate gamma function recursively.

    References
    ----------
    https://en.wikipedia.org/wiki/Multivariate_gamma_function
    """
    x = as_tensor(x)
    _dims = tf.range(p, dtype=x.dtype)
    p = as_tensor(p)
    return tf.add(0.25 * p * (p - 1.0) * LOGPI,
                  tf.reduce_sum(tf.lgamma(x[..., None] - 0.5 * _dims), axis=-1), name=name)


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


class capture_stdstream:
    """
    Capture a std stream such as stdout or stderr.

    Parameters
    ----------
    stream : str
        name of the stream to capture.
    forward : bool
        whether to forward the content to the original stream.
    """
    def __init__(self, stream, forward=True):
        self._stream = stream
        self._str_stream = io.StringIO()
        self._sys_stream = None
        self._forward = forward

    def __enter__(self):
        self._sys_stream = getattr(sys, self._stream)
        setattr(sys, self._stream, self)
        return self

    def __exit__(self, *args):
        setattr(sys, self._stream, self._sys_stream)
        self._sys_stream = None

    def write(self, *args, **kwargs):
        self._str_stream.write(*args, **kwargs)
        if self._forward and self._sys_stream:
            self._sys_stream.write(*args, **kwargs)

    @property
    def value(self):
        """
        str : content written to the std stream.
        """
        return self._str_stream.getvalue()
