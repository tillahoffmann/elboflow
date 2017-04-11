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

    x = as_tensor(x)
    if isinstance(statistic, numbers.Number):
        return x ** statistic
    elif statistic == 'entropy':
        return tf.constant(0.0)
    elif statistic == 'outer':
        return x[..., None, :] * x[..., :, None]
    elif statistic == 'log_det':
        return tf.log(tf.matrix_determinant(x))
    elif statistic == 'log':
        return tf.log(x)
    else:
        raise KeyError("'%s' is not a recognized statistic" % statistic)


class Distribution(BaseDistribution):
    """
    Base class for distributions.

    Parameters
    ----------
    parameters : list[str]
        parameter names
    """
    def __init__(self, parameters):
        self._statistics = {}
        self._parameters = parameters

    def _statistic(self, statistic):
        if statistic == 'outer':
            mean = self.statistic(1)
            return mean[..., :, None] * mean[..., None, :] + tf.diag(self.statistic('var'))
        elif statistic == 'std':
            return tf.sqrt(self.statistic('var'))
        else:
            raise KeyError("'%s' is not a recognized statistic for `%s`" %
                           (statistic, self.__class__.__name__))

    def _log_pdf(self, x):
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

    def log_pdf(self, x, reduce=False):
        """
        Evaluate the log of the distribution.

        Parameters
        ----------
        x : tf.Tensor or Distribution
            point at which to evaluate the log pdf
        """
        _log_pdf = self._log_pdf(x)
        if reduce:
            _log_pdf = tf.reduce_sum(_log_pdf)
        return _log_pdf

    @property
    def parameters(self):
        """dict[str, tf.Tensor] : parameters keyed by name"""
        return {name: getattr(self, '_%s' % name) for name in self._parameters}

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join(["%s=%s" % kv for kv in self.parameters.items()]))

    @property
    def mean(self):
        return self.statistic(1)

    @property
    def var(self):
        return self.statistic('var')

    @property
    def std(self):
        return self.statistic('std')

    @property
    def cov(self):
        return self.statistic('cov')

    @property
    def entropy(self):
        return self.statistic('entropy')


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
        super(NormalDistribution, self).__init__(['mean', 'precision'])
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
            return super(NormalDistribution, self)._statistic(statistic)

    def _log_pdf(self, x):
        chi2 = tf.square(self._mean) - 2.0 * self._mean * evaluate_statistic(x, 1) + \
            evaluate_statistic(x, 2)
        return 0.5 * (evaluate_statistic(self._precision, 'log') - LOG2PI) - \
            0.5 * evaluate_statistic(self._precision, 1) * chi2

    @staticmethod
    def linear_log_likelihood(y, x, theta, tau, reduce=False):
        """
        Evaluate the log likelihood of the observation `y` given features `x`, coefficients `theta`,
        and noise precision `tau`.

        Parameters
        ----------
        reduce : bool
            whether to aggregate the likelihood
        """
        y, x, theta, tau = map(as_tensor, [y, x, theta, tau])
        # Evaluate the pointwise expected log-likelihood
        chi2 = evaluate_statistic(y, 2) - 2.0 * tf.reduce_sum(
            evaluate_statistic(y[..., None], 1) * evaluate_statistic(x, 1) *
            evaluate_statistic(theta, 1), axis=-1
        ) + tf.reduce_sum(
            evaluate_statistic(x, 'outer') * evaluate_statistic(theta, 'outer'), axis=(-1, -2)
        )
        ll = 0.5 * (
            evaluate_statistic(tau, 'log') - LOG2PI - evaluate_statistic(tau, 1) * chi2
        )
        # Reduce the likelihood if desired
        if reduce:
            ll = tf.reduce_sum(ll)
        return ll


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
        super(GammaDistribution, self).__init__(['shape', 'scale'])
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
        elif statistic == 'log':
            return tf.digamma(self._shape) - tf.lgamma(self._scale)
        else:
            return super(GammaDistribution, self)._statistic(statistic)

    def _log_pdf(self, x):
        return - tf.lgamma(self._shape) + self._shape * tf.log(self._scale) + \
            (self._shape - 1.0) * evaluate_statistic(x, 'log') - \
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
        super(CategoricalDistribution, self).__init__(['p'])
        self._p = as_tensor(p)

    def _statistic(self, statistic):
        if statistic == 'entropy':
            return - tf.reduce_sum(self._p * tf.log(self._p), axis=-1)
        elif statistic in (1, 2):
            return self._p
        elif statistic == 'var':
            return self._p * (1.0 - self._p)
        else:
            return super(CategoricalDistribution, self)._statistic(statistic)

    def _log_pdf(self, x):
        return tf.reduce_sum(x * tf.log(self._p), axis=-1)


class DirichletDistribution(Distribution):
    """
    Dirichlet distribution.

    Parameters
    ----------
    alpha : tf.Tensor
        concentration parameter
    """
    def __init__(self, alpha):
        super(DirichletDistribution, self).__init__(['alpha'])
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
            return super(DirichletDistribution, self)._statistic(statistic)

    def _log_pdf(self, x):
        return tf.reduce_sum((self._alpha - 1.0) * evaluate_statistic(x, 'log'), axis=-1) - \
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
        super(MultiNormalDistribution, self).__init__(['mean', 'precision'])
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
            return super(MultiNormalDistribution, self)._statistic(statistic)

    def _log_pdf(self, x):
        x_mean = evaluate_statistic(x, 1)
        arg = evaluate_statistic(x, 'outer') + self.statistic('outer_mean') - \
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
        super(WishartDistribution, self).__init__(['shape', 'scale'])
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
            return super(WishartDistribution, self)._statistic(statistic)

    def _log_pdf(self, x):
        x_logdet = evaluate_statistic(x, 'log_det')
        x_mean = evaluate_statistic(x, 1)
        p = self.statistic('ndim')

        return 0.5 * x_logdet * (self._shape - p - 1.0) - \
            0.5 * tf.reduce_sum(self._scale * x_mean, axis=(-1, -2)) - \
            0.5 * self._shape * p * LOG2 - lmultigamma(0.5 * self._shape, p) + \
            0.5 * self._shape * tf.log(tf.matrix_determinant(self._scale))
