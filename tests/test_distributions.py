import itertools as it
import numbers
import scipy.stats
import pytest
import numpy as np
import elboflow as ef
import tensorflow as tf


def _scipy_var(dist):
    if hasattr(dist, 'var'):
        return dist.var()
    elif hasattr(dist, 'cov'):
        return np.diag(dist.cov() if callable(dist.cov) else dist.cov)
    else:
        raise NotImplementedError("%s does not support 'var'" % dist)


def _scipy_mean(dist):
    return dist.mean() if callable(dist.mean) else dist.mean


@pytest.fixture(params=it.chain(
    [(ef.NormalDistribution(mean, precision), scipy.stats.norm(mean, 1 / np.sqrt(precision)))
     for mean, precision in [(0, 1), (3, 0.1), (-5, 2)]],
    [(ef.GammaDistribution(shape, scale), scipy.stats.gamma(shape, scale=1 / scale))
     for shape, scale in [(1e-3, 3), (50, 2)]],
    [(ef.DirichletDistribution(alpha), scipy.stats.dirichlet(alpha)) for alpha in
     [np.ones(5), np.random.gamma(1, 1, 7)]],
    [(ef.WishartDistribution(shape, scale), scipy.stats.wishart(shape, scale))
     for shape, scale in [(4, np.eye(2))]],
    [(ef.CategoricalDistribution(p), scipy.stats.multinomial(1, p)) for p in
     [np.ones(3) / 3, np.random.dirichlet(np.ones(7))]],
    [(ef.MultiNormalDistribution(mean, precision),
      scipy.stats.multivariate_normal(mean, np.linalg.inv(precision))) for mean, precision
     in [(np.zeros(2), np.eye(2)), ([-3, 2], [[4, -1], [-1, 2]])]],
))
def distribution_pair(request):
    return request.param


@pytest.mark.parametrize('statistic', [1, 2, 'var', 'entropy'])
def test_statistic(session, distribution_pair, statistic):
    # Get the statistic from the tensorflow object
    ef_dist, scipy_dist = distribution_pair
    actual = session.run(ef_dist.statistic(statistic))

    # Get the statistic from scipy.stats
    if isinstance(statistic, numbers.Number):
        if hasattr(scipy_dist, 'moment'):
            desired = scipy_dist.moment(statistic)
        elif statistic == 1:
            desired = _scipy_mean(scipy_dist)
        elif statistic == 2:
            desired = _scipy_mean(scipy_dist) ** 2 + _scipy_var(scipy_dist)
        else:
            raise KeyError(statistic)
    elif statistic == 'var':
        desired = _scipy_var(scipy_dist)
    elif statistic == 'entropy':
        desired = scipy_dist.entropy()
    else:
        raise KeyError(statistic)

    np.testing.assert_allclose(actual, desired, 1e-5)


def test_log_pdf(session, distribution_pair):
    ef_dist, scipy_dist = distribution_pair
    ef_x = scipy_x = scipy_dist.rvs()

    if isinstance(ef_dist, ef.DirichletDistribution):
        scipy_x = scipy_x[0]

    actual = session.run(ef_dist.log_pdf(ef_x))
    if hasattr(scipy_dist, 'logpdf'):
        desired = scipy_dist.logpdf(scipy_x)
    elif hasattr(scipy_dist, 'logpmf'):
        desired = scipy_dist.logpmf(scipy_x)
    else:
        raise NotImplementedError("%s does not support 'log*'" % scipy_dist)
    np.testing.assert_allclose(actual, desired, 1e-5)
