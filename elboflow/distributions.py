import numbers
import tensorflow as tf

from .util import *


def evaluate_statistic(x, statistic):
    """
    Evaluate statistic of a value or distribution.

    Parameters
    ----------
    x : tf.Tensor or Distribution
        value or distribution
    statistic : int, str or callable
        statistic to evaluate
    """
    if isinstance(x, Distribution):
        return x.statistic(statistic)
    elif callable(statistic):
        return statistic(x)
    elif isinstance(statistic, numbers.Number):
        return x ** statistic
    elif statistic == 'entropy':
        return tf.constant(0.0)
    else:
        return KeyError("'%s' is not a recognized statistic" % statistic)


class Distribution:
    """
    Base class for distributions.
    """
    def __init__(self):
        self._statistics = {}

    def _statistic(self, statistic):
        raise NotImplementedError

    def statistic(self, statistic):
        """
        Evaluate a statistic of the distribution.

        Parameters
        ----------
        statistic : str, int, or callable
            statistic to evaluate
        """
        # Get the statistic from the cache
        _statistic = self._statistics.get(statistic)
        if _statistic is None:
            _statistic = self._statistic(statistic)
            # Save the statistic in the cache
            self._statistics[statistic] = _statistic
        return _statistic

    def log_pdf(self, x):
        """
        Evaluate the log of the distribution.

        Parameters
        ----------
        x : tf.Tensor or Distribution
            point at which to evaluate the log pdf
        """
        raise NotImplementedError


class NormalDistribution(Distribution):
    """
    Univariate normal distribution.

    Parameters
    ----------
    mean : tf.Tensor
        mean of the normal distribution
    precision : tf.Tensor
        precision of the normal distribution
    """
    def __init__(self, mean, precision):
        super(NormalDistribution, self).__init__()
        self._mean = as_tensor(mean)
        self._precision = as_tensor(precision)

    def _statistic(self, statistic):
        if statistic == 'entropy':
            return 0.5 * (LOG2PIE - tf.log(self._precision))
        elif statistic in ('mode', 1):
            return self._mean
        elif statistic == 'var':
            return tf.reciprocal(self._precision)
        elif statistic == 2:
            return tf.square(self.statistic(1)) + self.statistic('var')
        else:
            raise KeyError(statistic)

    def log_pdf(self, x):
        chi2 = tf.square(self._mean) - 2.0 * self._mean * evaluate_statistic(x, 1) + \
            evaluate_statistic(x, 2)
        return 0.5 * (evaluate_statistic(self._precision, tf.log) - LOG2PI) - \
            0.5 * evaluate_statistic(self._precision, 1) * chi2


class GammaDistribution(Distribution):
    """
    Univariate gamma distribution.

    Parameters
    ----------
    shape : tf.Tensor
        shape parameter
    scale : tf.Tensor
        scale parameter
    """
    def __init__(self, shape, scale):
        super(GammaDistribution, self).__init__()
        self._shape = as_tensor(shape)
        self._scale = as_tensor(scale)

    def _statistic(self, statistic):
        if statistic == 'entropy':
            return self._shape - tf.log(self._scale) + tf.lgamma(self._shape) + \
                (1.0 - self._shape) * tf.digamma(self._shape)
        elif statistic == 1:
            return self._shape / self._scale
        elif statistic == 'mode':
            raise (self._shape - 1.0) / self._scale
        elif statistic == 'var':
            return self._shape / np.square(self._scale)
        elif statistic == 2:
            return self._shape * (self._shape + 1.0) / np.square(self._scale)
        else:
            raise KeyError

    def log_pdf(self, x):
        return - tf.lgamma(self._shape) + self._shape * tf.log(self._scale) + \
            (self._shape - 1.0) * evaluate_statistic(x, tf.log) - \
            self._scale * evaluate_statistic(x, 1)


class CategoricalDistribution(Distribution):
    """
    Univariate categorical distribution.

    Parameters
    ----------
    p : tf.Tensor
        tensor of probabilities
    """
    def __init__(self, p):
        super(CategoricalDistribution, self).__init__()
        self._p = as_tensor(p)

    def _statistic(self, statistic):
        if statistic == 'entropy':
            return - tf.reduce_sum(self._p * tf.log(self._p), axis=-1)
        elif statistic in (1, 2):
            return self._p
        elif statistic == 'var':
            return self._p * (1.0 - self._p)
        else:
            raise KeyError

    def log_pdf(self, x):
        raise NotImplementedError


class DirichletDistribution(Distribution):
    """
    Dirichlet distribution.

    Parameters
    ----------
    alpha : tf.Tensor
        concentration parameter

    TODO: fix shapes with keeping or not keeping the dimensions
    """
    def __init__(self, alpha):
        super(DirichletDistribution, self).__init__()
        self._alpha = as_tensor(alpha)

    def _statistic(self, statistic):
        if statistic == 'alpha_total':
            return tf.reduce_sum(self._alpha, -1, True, 'alpha_total')
        elif statistic == 'alpha_totalm1':
            return tf.reduce_sum(self._alpha - 1.0, -1, True, 'alpha_totalm1')
        elif statistic == 1:
            return tf.divide(self._alpha, self.statistic('alpha_total'), 'mean')
        elif statistic == 'mode':
            return (self._alpha - 1) / self.statistic('alpha_totalm1')
        elif statistic == 'var':
            alpha_total = self.statistic('alpha_total')
            return self._alpha * (alpha_total - self._alpha) / \
                (tf.square(alpha_total) * (alpha_total + 1.0))
        elif statistic == 2:
            return tf.square(self.statistic(1)) + self.statistic('var')
        elif statistic == 'log_normalization':
            return tf.reduce_sum(tf.lgamma(self._alpha), -1) - \
                tf.lgamma(self.statistic('alpha_total')[..., 0])
        elif statistic == 'entropy':
            return self.statistic('log_normalization') + self.statistic('alpha_totalm1')[..., 0] * \
                tf.digamma(self.statistic('alpha_total')[..., 0]) - \
                tf.reduce_sum((self._alpha - 1.0) * tf.digamma(self._alpha), -1)
        else:
            raise KeyError

    def log_pdf(self, x):
        return (self._alpha - 1.0) * evaluate_statistic(x, tf.log) - \
            self.statistic('log_normalization')


class MultiNormalDistribution(Distribution):
    """
    Multivariate normal distribution.

    Parameters
    ----------
    mean : tf.Tensor
        mean of the multivariate normal distribution
    precision : tf.Tensor
        precision of the multivariate normal distribution
    """
    def __init__(self, mean, precision):
        super(MultiNormalDistribution, self).__init__()
        self._mean = as_tensor(mean)
        self._precision = as_tensor(precision)

    def _statistic(self, statistic):
        if statistic == 1:
            return self._mean
        elif statistic == 'cov':
            return tf.matrix_inverse(self._precision)
        elif statistic == 'var':
            return tf.matrix_diag_part(self.statistic('cov'))
        elif statistic == 'outer':
            return self.statistic('outer_mean') + self.statistic('cov')
        elif statistic == 'outer_mean':
            return self._mean[..., None, :] * self._mean[..., :, None]
        elif statistic == 2:
            return tf.square(self._mean) + self.statistic('var')
        elif statistic == 'entropy':
            return 0.5 * (LOG2PIE * self.statistic('ndim') -
                          self.statistic('log_det_precision'))
        elif statistic == 'ndim':
            return tf.to_float(self._mean.get_shape()[-1])
        elif statistic == 'log_det_precision':
            return tf.log(tf.matrix_determinant(self._precision))
        else:
            raise KeyError

    def log_pdf(self, x):
        x_mean = evaluate_statistic(x, 'mean')
        arg = evaluate_statistic(x, 'outer') + \
            self.statistic('outer_mean') - \
            x_mean[..., None, :] * self._mean[..., :, None] - \
            x_mean[..., :, None] * self._mean[..., None, :]

        return - 0.5 * self.statistic('ndim') * LOG2PI + \
            0.5 * self.statistic('log_det_precision') - \
            0.5 * tf.reduce_sum(self._precision * arg, axis=(-1, -2))


class WishartDistribution(Distribution):
    """
    Wishart distribution.

    Parameters
    ----------
    shape : array_like
        shape parameter or degrees of freedom which must be greater than `ndim - 1`
    scale : array_like
        scale matrix
    """
    def __init__(self, shape, scale):
        super(WishartDistribution, self).__init__()
        self._shape = as_tensor(shape)
        self._scale = as_tensor(scale)

    def _statistic(self, statistic):
        if statistic == 1:
            return self._shape[..., None, None] * self.statistic('inv_scale')
        elif statistic == 'inv_scale':
            return tf.matrix_inverse(self._scale)
        elif statistic == 'var':
            inv_scale = self.statistic('inv_scale')
            diag = tf.matrix_diag_part(inv_scale)
            return self._shape * (tf.square(inv_scale) + diag[..., None, :] * diag[..., :, None])
        elif statistic == 2:
            return tf.square(self.statistic(1)) + self.statistic('var')
        elif statistic == 'ndim':
            return tf.to_float(self._scale.get_shape()[-1])
        elif statistic == 'entropy':
            p = self.statistic('ndim')
            return -0.5 * (p + 1.0) * tf.log(tf.matrix_determinant(self._scale)) + \
                0.5 * p * (p + 1.0) * LOG2 + lmultigamma(0.5 * self._shape, p) - \
                0.5 * (self._shape - p - 1.0) * multidigamma(0.5 * self._shape, p) + \
                0.5 * self._shape * p
        else:
            raise KeyError(statistic)

    def log_pdf(self, x):
        x_logdet = evaluate_statistic(x, 'log_det')
        x_mean = evaluate_statistic(x, 1)
        p = self.statistic('ndim')

        return 0.5 * x_logdet * (self._shape - p - 1.0) - \
            0.5 * tf.reduce_sum(self._scale * x_mean, axis=(-1, -2)) - \
            0.5 * self._shape * p * LOG2 - lmultigamma(0.5 * self._shape, p) + \
            0.5 * self._shape * tf.log(tf.matrix_determinant(self._scale))
