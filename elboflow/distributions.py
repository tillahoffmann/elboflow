import numbers
import tensorflow as tf

from .util import *


def evaluate_statistic(x, statistic, name=None):
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
        return tf.pow(x, statistic, name)
    elif statistic == 'entropy':
        return tf.constant(0.0, name)
    elif statistic == 'outer':
        return tf.multiply(x[..., None, :], x[..., :, None], name)
    elif statistic == 'log_det':
        return tf.log(tf.matrix_determinant(x), name)
    elif statistic == 'log':
        return tf.log(x, name)
    elif statistic == 'log1m':
        return tf.log(1.0 - x, name)
    elif statistic == 'lgamma':
        return tf.lgamma(x, name)
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
    def __init__(self, parameters, supported_statistics=None):
        self._statistics = {}
        self._parameters = parameters
        self.supported_statistics = [1, 2, 'var', 'entropy']
        if supported_statistics:
            self.supported_statistics.extend(supported_statistics)

    def _statistic(self, statistic, name):
        if statistic == 'outer':
            mean = self.statistic(1)
            return tf.add(mean[..., :, None] * mean[..., None, :], self.statistic('cov'), name)
        elif statistic == 'cov':
            return tf.diag(self.statistic('var'), name)
        elif statistic == 'std':
            return tf.sqrt(self.statistic('var'), name)
        else:
            raise KeyError("'%s' is not a recognized statistic for `%s`" %
                           (statistic, self.__class__.__name__))

    def statistic(self, statistic, name=None):
        """
        Evaluate a statistic of the distribution.

        Parameters
        ----------
        statistic : str, int, or callable
            statistic to evaluate
        name : str or None
            name of the resulting operation
        """
        # Get the statistic from the cache
        _statistic = self._statistics.get(statistic)
        if _statistic is None:
            _statistic = self._statistic(statistic, name)
            # Save the statistic in the cache
            self._statistics[statistic] = _statistic
        return _statistic

    def log_proba(self, x, name=None):
        """
        Evaluate the log of the probability distribution evaluated at `x`.

        Parameters
        ----------
        x : tf.Tensor or Distribution
            point at which to evaluate the log pdf
        name : str or None
            name of the resulting operation
        """
        return self.log_likelihood(x, **self.parameters, name=name)

    @property
    def parameters(self):
        """dict[str, tf.Tensor] : parameters keyed by name"""
        return {name: getattr(self, '_%s' % name) for name in self._parameters}

    def to_str(self, session=None):
        parameters = self.parameters
        if session:
            parameters = {key: session.run(value) for key, value in parameters.items()}
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join(["%s=%s" % kv for kv in parameters.items()]))

    def __str__(self):
        return self.to_str()

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

    @staticmethod
    def log_likelihood(x, *parameters, name=None):
        """
        Evaluate the expected log likelihood for `x` given `parameters`.

        Parameters
        ----------
        x : tf.Tensor, or Distribution
            value or distribution at which to evaluate the expected log likelihood
        name : str or None
            name of the resulting operation
        """
        raise NotImplementedError

    @property
    def sample_rank(self):
        """int : rank of samples (0 for scalars, 1 for vectors, 2 for matrices, etc.)"""
        raise NotImplementedError

    @property
    def batch_shape(self):
        """tf.Tensor : shape of batches of samples"""
        return self.shape[:-self.sample_rank]

    @property
    def sample_shape(self):
        """tf.Tensor : shape of samples"""
        if self.sample_rank > 0:
            return self.shape[-self.sample_rank:]
        else:
            return as_tensor([], tf.int32)

    @property
    def shape(self):
        """tf.Tensor : shape of batches and samples"""
        return tf.shape(self.mean)


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

    def _statistic(self, statistic, name):
        if statistic == 'entropy':
            return tf.multiply(0.5, LOG2PIE - tf.log(self._precision), name)
        elif statistic == 1:
            return self._mean
        elif statistic == 'var':
            return tf.reciprocal(self._precision, name)
        elif statistic == 2:
            return tf.add(tf.square(self.statistic(1)), self.statistic('var'), name)
        else:
            return super(NormalDistribution, self)._statistic(statistic, name)

    @property
    def sample_rank(self):
        return 0

    @staticmethod
    def log_likelihood(x, mean, precision, name=None):  # pylint: disable=W0221
        chi2 = evaluate_statistic(x, 2) - 2 * evaluate_statistic(x, 1) * \
            evaluate_statistic(mean, 1) + evaluate_statistic(mean, 2)
        return tf.multiply(0.5, evaluate_statistic(precision, 'log') - LOG2PI -
                           evaluate_statistic(precision, 1) * chi2, name)

    @staticmethod
    def linear_log_likelihood(x, y, theta, tau, name=None):
        """
        Evaluate the log likelihood of the observation `y` given features `x`, coefficients `theta`,
        and noise precision `tau`.

        Parameters
        ----------
        TODO: extend parameter documentation
        """
        y, x, theta, tau = map(as_tensor, [y, x, theta, tau])
        # Evaluate the pointwise expected log-likelihood
        chi2 = evaluate_statistic(y, 2) - 2.0 * tf.reduce_sum(
            evaluate_statistic(y[..., None], 1) * evaluate_statistic(x, 1) *
            evaluate_statistic(theta, 1), axis=-1
        ) + tf.reduce_sum(
            evaluate_statistic(x, 'outer') * evaluate_statistic(theta, 'outer'), axis=(-1, -2)
        )
        ll = tf.multiply(0.5, evaluate_statistic(tau, 'log') - LOG2PI -
                         evaluate_statistic(tau, 1) * chi2, name)
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
        super(GammaDistribution, self).__init__(['shape', 'scale'], ['log'])
        self._shape = as_tensor(shape)
        self._scale = as_tensor(scale)

    def _statistic(self, statistic, name):
        if statistic == 'entropy':
            return tf.add(self._shape - tf.log(self._scale) + tf.lgamma(self._shape),
                          (1.0 - self._shape) * tf.digamma(self._shape), name)
        elif statistic == 1:
            return tf.divide(self._shape, self._scale, name)
        elif statistic == 2:
            return tf.divide(self._shape * (self._shape + 1.0), np.square(self._scale), name)
        elif statistic == 'var':
            return tf.divide(self._shape, np.square(self._scale), name)
        elif statistic == 'log':
            return tf.subtract(tf.digamma(self._shape), tf.log(self._scale), name)
        else:
            return super(GammaDistribution, self)._statistic(statistic, name)

    @property
    def sample_rank(self):
        return 0

    @staticmethod
    def log_likelihood(x, shape, scale, name=None):  # pylint: disable=W0221
        shape_1 = evaluate_statistic(shape, 1)
        return tf.subtract(shape_1 * evaluate_statistic(scale, 'log') + (shape_1 - 1.0) *
                           evaluate_statistic(x, 'log') - evaluate_statistic(scale, 1) *
                           evaluate_statistic(x, 1), evaluate_statistic(shape, 'lgamma'), name)


class CategoricalDistribution(Distribution):
    """
    Univariate categorical distribution.

    Parameters
    ----------
    p : tf.Tensor
        tensor of probabilities
    """
    def __init__(self, p):
        super(CategoricalDistribution, self).__init__(['p'], ['cov', 'outer'])
        self._p = as_tensor(p)

    def _statistic(self, statistic, name):
        if statistic == 'entropy':
            return - tf.reduce_sum(self._p * tf.log(self._p), axis=-1, name=name)
        elif isinstance(statistic, numbers.Real) and statistic > 0:
            return self._p
        elif statistic == 'var':
            return tf.multiply(self._p, (1.0 - self._p), name)
        elif statistic == 'cov':
            cov = - self._p[..., None, :] * self._p[..., :, None]
            return tf.matrix_set_diag(cov, self.statistic('var'), name)
        elif statistic == 'outer':
            return tf.matrix_diag(self._p, name)
        else:
            return super(CategoricalDistribution, self)._statistic(statistic, name)

    @property
    def sample_rank(self):
        return 1

    @staticmethod
    def log_likelihood(x, p, name=None):  # pylint: disable=W0221
        return tf.reduce_sum(evaluate_statistic(x, 1) * evaluate_statistic(p, 'log'), axis=-1,
                             name=name)

    @staticmethod
    def mixture_log_likelihood(z, expected_log_likelihood, name=None):
        """
        Evaluate the expected log likelihood of a mixture distribution given indicators `z`.

        Parameters
        ----------
        z : tf.Tensor
            indiciator variables of shape `(..., k)` where `k` is the number of mixture components
        expected_log_likelihood : tf.Tensor
            expected log likelihood given component membership of the same shape as `z`
        """
        return tf.reduce_sum(evaluate_statistic(z, 1) * expected_log_likelihood, axis=-1, name=name)

    @staticmethod
    def interacting_mixture_log_likelihood(z, expected_log_likelihood, name=None):
        """
        Evaluate the expected log likelihood of a mixture distribution with interactions given
        indicators `z`.

        Parameters
        ----------
        z : tf.Tensor
            indiciator variables of shape `(n, k)`, where `n` is the number of observations and `k`
            is the number of mixture components
        expected_log_likelihood : tf.Tensor
            expected log likelihood given component membership with shape `(n, n, k, k)` such that
            the `(i, j, a, b)` element corresponds to the log likelihood term for the interaction
            between entities `i` and `j` given that they belong to components `a` and `b`,
            respectively.
        """
        # Get the mean
        z_1 = evaluate_statistic(z, 1)
        shape = z_1.shape
        if len(shape) != 2:
            raise ValueError("interacting mixtures are only supported for vectors of observations, "
                             "i.e. the shape of the indicators must be `(num_observations, "
                             "num_components)`")
        # Create the product of means with shape (num_obs, num_obs, num_comps, num_comps)
        zz = z_1[:, None, :, None] * z_1[None, :, None, :]
        # Add covariance terms to the diagonal (wrt observations) to account for repeated indices
        # (distinct indices are independent by assumption)
        z_cov = evaluate_statistic(z, 'cov')
        zz += tf.eye(shape[0].value)[..., None, None] * z_cov

        return tf.reduce_sum(zz * expected_log_likelihood, axis=(-1, -2), name=name)


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

    def _statistic(self, statistic, name):
        if statistic == '_alpha_total':
            return tf.reduce_sum(self._alpha, -1, True, name=name)
        elif statistic == '_alpha_totalm1':
            return tf.reduce_sum(self._alpha - 1.0, -1, True, name=name)
        elif statistic == 1:
            return tf.divide(self._alpha, self.statistic('_alpha_total'), name)
        elif statistic == 2:
            return tf.add(tf.square(self.statistic(1)), self.statistic('var'), name)
        elif statistic == 'var':
            _alpha_total = self.statistic('_alpha_total')
            return tf.divide(self._alpha * (_alpha_total - self._alpha),
                             tf.square(_alpha_total) * (_alpha_total + 1.0), name)
        elif statistic == 'log':
            return tf.subtract(tf.digamma(self._alpha), tf.digamma(self.statistic('_alpha_total')),
                               name)
        elif statistic == '_log_normalization':
            return tf.subtract(tf.reduce_sum(tf.lgamma(self._alpha), -1),
                               tf.lgamma(self.statistic('_alpha_total')[..., 0]), name)
        elif statistic == 'entropy':
            return tf.add(self.statistic('_log_normalization'),
                          self.statistic('_alpha_totalm1')[..., 0] *
                          tf.digamma(self.statistic('_alpha_total')[..., 0]) -
                          tf.reduce_sum((self._alpha - 1.0) * tf.digamma(self._alpha), -1), name)
        else:
            return super(DirichletDistribution, self)._statistic(statistic, name)

    @property
    def sample_rank(self):
        return 1

    @staticmethod
    def log_likelihood(x, alpha, name=None):  # pylint: disable=W0221
        assert_constant(alpha)
        return tf.add(tf.reduce_sum((alpha - 1.0) * evaluate_statistic(x, 'log'), axis=-1) -
                      tf.reduce_sum(tf.lgamma(alpha), -1), tf.lgamma(tf.reduce_sum(alpha, -1)),
                      name)


class BetaDistribution(Distribution):
    """
    Beta distribution.

    Parameters
    ----------
    a : tf.Tensor
        first shape parameter
    b : tf.Tensor
        second shape parameter
    """
    def __init__(self, a, b):
        super(BetaDistribution, self).__init__(['a', 'b'])
        self._a = as_tensor(a)
        self._b = as_tensor(b)

    def _statistic(self, statistic, name):
        if statistic == 1:
            return tf.divide(self._a, self.statistic('_total'), name)
        elif statistic == 2:
            _total = self.statistic('_total')
            return tf.divide(self._a * (self._a + 1.0), (_total + 1.0) * _total, name)
        elif statistic == 'var':
            _total = self.statistic('_total')
            return tf.divide(self._a * self._b, tf.square(_total) * (_total + 1.0), name)
        elif statistic == 'log':
            return tf.subtract(tf.digamma(self._a), tf.digamma(self.statistic('_total')), name)
        elif statistic == 'log1m':
            return tf.subtract(tf.digamma(self._b), tf.digamma(self.statistic('_total')), name)
        elif statistic == '_total':
            return tf.add(self._a, self._b, name)
        elif statistic == 'entropy':
            _total = self.statistic('_total')
            return tf.add(tf.lgamma(self._a), tf.lgamma(self._b) - tf.lgamma(_total) +
                          (1.0 - self._a) * tf.digamma(self._a) + (1.0 - self._b) *
                          tf.digamma(self._b) + (_total - 2.0) * tf.digamma(_total), name)
        else:
            return super(BetaDistribution, self)._statistic(statistic, name)

    @property
    def sample_rank(self):
        return 0

    @staticmethod
    def log_likelihood(x, a, b, name=None):  # pylint: disable=W0221
        assert_constant(a)
        assert_constant(b)

        x_log = evaluate_statistic(x, 'log')
        x_log1m = evaluate_statistic(x, 'log1m')
        return tf.add((a - 1.0) * x_log + (b - 1.0) * x_log1m,
                      tf.lgamma(a + b) - tf.lgamma(a) - tf.lgamma(b), name)


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
        super(MultiNormalDistribution, self).__init__(['mean', 'precision'], ['cov', 'outer'])
        self._mean = as_tensor(mean)
        self._precision = as_tensor(precision)

    def _statistic(self, statistic, name):
        if statistic == 1:
            return self._mean
        elif statistic == 2:
            return tf.add(tf.square(self._mean), self.statistic('var'), name)
        elif statistic == 'cov':
            return tf.matrix_inverse(self._precision, name=name)
        elif statistic == 'var':
            return tf.matrix_diag_part(self.statistic('cov'), name)
        elif statistic == 'outer':
            return tf.add(self._mean[..., None, :] * self._mean[..., :, None],
                          self.statistic('cov'), name)
        elif statistic == 'entropy':
            return tf.multiply(0.5, LOG2PIE * self.statistic('_ndim') -
                               self.statistic('_log_det_precision'), name)
        elif statistic == '_ndim':
            return tf.to_float(self.sample_shape[-1], name)
        elif statistic == '_log_det_precision':
            return tf.log(tf.matrix_determinant(self._precision), name)
        else:
            return super(MultiNormalDistribution, self)._statistic(statistic, name)

    @property
    def sample_rank(self):
        return 1

    @staticmethod
    def log_likelihood(x, mean, precision, name=None):  # pylint: disable=W0221
        x_1 = evaluate_statistic(x, 1)
        mean_1 = evaluate_statistic(mean, 1)
        arg = evaluate_statistic(x, 'outer') + evaluate_statistic(mean, 'outer') - \
            x_1[..., None, :] * mean_1[..., :, None] - \
            x_1[..., :, None] * mean_1[..., None, :]

        ndim = tf.to_float(tf.shape(mean_1)[-1])

        return tf.multiply(0.5, - ndim * LOG2PI + evaluate_statistic(precision, 'log_det') -
                           tf.reduce_sum(evaluate_statistic(precision, 1) * arg, axis=(-1, -2)),
                           name)


class WishartDistribution(Distribution):
    """
    Wishart distribution.

    Parameters
    ----------
    shape : array_like
        shape parameter or degrees of freedom which must be greater than `ndim - 1`
    scale : array_like
        scale matrix

    Notes
    -----
    We use an unconventional parametrization of the Wishart distribution in contrast to the standard
    approach used, e.g. on [wikipedia](https://en.wikipedia.org/wiki/Wishart_distribution). In
    particular, our scale parameter differs by a matrix inverse to be consistent with the
    parametrization of the gamma distribution above.
    """
    def __init__(self, shape, scale):
        super(WishartDistribution, self).__init__(['shape', 'scale'], ['log_det'])
        self._shape = as_tensor(shape)
        self._scale = as_tensor(scale)

    def _statistic(self, statistic, name):
        if statistic == 1:
            return tf.multiply(self._shape[..., None, None], self.statistic('_inv_scale'), name)
        elif statistic == '_inv_scale':
            return tf.matrix_inverse(self._scale, name=name)
        elif statistic == 'var':
            inv_scale = self.statistic('_inv_scale')
            diag = tf.matrix_diag_part(inv_scale)
            return tf.multiply(self._shape, tf.square(inv_scale) + diag[..., None, :] *
                               diag[..., :, None], name)
        elif statistic == 2:
            return tf.add(tf.square(self.statistic(1)), self.statistic('var'), name)
        elif statistic == '_ndim':
            return tf.to_float(self.sample_shape[-1], name)
        elif statistic == 'log_det':
            p = self.statistic('_ndim')
            return tf.subtract(multidigamma(0.5 * self._shape, p) + p * LOG2,
                               tf.log(tf.matrix_determinant(self._scale)), name)
        elif statistic == 'entropy':
            p = self.statistic('_ndim')
            return tf.subtract(0.5 * p * (p + 1.0) * LOG2 + lmultigamma(0.5 * self._shape, p) -
                               0.5 * (self._shape - p - 1.0) * multidigamma(0.5 * self._shape, p) +
                               0.5 * self._shape * p, 0.5 * (p + 1.0) *
                               tf.log(tf.matrix_determinant(self._scale)), name)
        else:
            return super(WishartDistribution, self)._statistic(statistic, name)

    @property
    def sample_rank(self):
        return 2

    @staticmethod
    def log_likelihood(x, shape, scale, name=None):  # pylint: disable=W0221
        assert_constant(shape)
        x_logdet = evaluate_statistic(x, 'log_det')
        x_1 = evaluate_statistic(x, 1)
        shape_1 = evaluate_statistic(shape, 1)
        scale_1 = evaluate_statistic(scale, 1)
        p = tf.to_float(tf.shape(scale_1)[-1])

        return tf.add(0.5 * x_logdet * (shape_1 - p - 1.0) - 0.5 *
                      tf.reduce_sum(scale_1 * x_1, axis=(-1, -2)) - 0.5 * shape_1 * p * LOG2 -
                      lmultigamma(0.5 * shape, p), 0.5 * shape_1 *
                      evaluate_statistic(scale, 'log_det'), name)
