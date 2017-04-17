import itertools as it
import numbers
import scipy.stats
import pytest
import numpy as np
import elboflow as ef


def evaluate_scipy_statistic(dist, statistic):
    # Compute the exact values
    if statistic == 'entropy':
        return dist.entropy()
    elif isinstance(statistic, numbers.Number):
        if hasattr(dist, 'moment'):
            return dist.moment(statistic)
        elif statistic == 1:
            return dist.mean() if callable(dist.mean) else dist.mean
        elif statistic == 2:
            return evaluate_scipy_statistic(dist, 1) ** 2 + evaluate_scipy_statistic(dist, 'var')
        else:
            raise KeyError(statistic)
    elif statistic == 'var':
        if hasattr(dist, 'var'):
            return dist.var()
        elif hasattr(dist, 'cov'):
            return np.diag(dist.cov() if callable(dist.cov) else dist.cov)
    elif statistic == 'cov':
        if hasattr(dist, 'cov'):
            return dist.cov() if callable(dist.cov) else dist.cov
    elif statistic == 'outer':
        mean = evaluate_scipy_statistic(dist, 1)
        cov = evaluate_scipy_statistic(dist, 'cov')
        return mean[..., None] * mean[..., None, :] + cov
    elif statistic in ('log', 'log_det'):
        num_samples = 1000
        x = dist.rvs(size=num_samples)
        if statistic == 'log':
            statistic = np.log(x)
        elif statistic == 'log_det':
            statistic = np.log(np.linalg.det(x))
        else:
            raise KeyError(statistic)
        return np.mean(statistic, axis=0), np.std(statistic, axis=0) / np.sqrt(num_samples - 1)

    raise RuntimeError("cannot evaluate statistic '%s' on %s" % (statistic, dist))


@pytest.fixture(params=it.chain(
    [(ef.NormalDistribution(mean, precision), scipy.stats.norm(mean, 1 / np.sqrt(precision)))
     for mean, precision in [(0, 1), (3, 0.1), (-5, 2)]],
    [(ef.GammaDistribution(shape, scale), scipy.stats.gamma(shape, scale=1 / scale))
     for shape, scale in [(.1, 3), (50, 2)]],
    [(ef.DirichletDistribution(alpha), scipy.stats.dirichlet(alpha)) for alpha in
     [np.ones(5), np.random.gamma(1, 1, 7)]],
    [(ef.WishartDistribution(shape, scale), scipy.stats.wishart(shape, np.linalg.inv(scale)))
     for shape, scale in [(4, 10 * np.eye(2)), (3, [(2, -1), (-1, 3)])]],
    [(ef.CategoricalDistribution(p), scipy.stats.multinomial(1, p)) for p in
     [np.ones(3) / 3, np.random.dirichlet(np.ones(7))]],
    [(ef.MultiNormalDistribution(mean, precision),
      scipy.stats.multivariate_normal(mean, np.linalg.inv(precision))) for mean, precision
     in [(np.zeros(2), np.eye(2)), ([-3, 2], [[4, -1], [-1, 2]])]],
))
def distribution_pair(request):
    return request.param


def test_statistic(session, distribution_pair):
    # Get the statistic from the tensorflow object
    ef_dist, scipy_dist = distribution_pair

    for statistic in ef_dist.supported_statistics:
        actual = session.run(ef.evaluate_statistic(ef_dist, statistic))
        desired = evaluate_scipy_statistic(scipy_dist, statistic)

        # Demand 3-sigma consistency if the statistic is sampled
        err_msg = "inconsistent statistic '%s' for `%s`: expected %%s but got %s" % \
            (statistic, ef_dist.to_str(session), actual)
        if isinstance(desired, tuple):
            desired_mean, desired_std = desired
            z_score = (actual - desired_mean) / desired_std
            np.testing.assert_array_less(np.abs(z_score), 3, err_msg % \
                ("%s +- %s" % (desired_mean, desired_std)))
        # Or demand close results
        else:
            np.testing.assert_allclose(actual, desired, 1e-5, err_msg=err_msg % desired)


def test_log_proba(session, distribution_pair):
    ef_dist, scipy_dist = distribution_pair
    ef_x = scipy_x = scipy_dist.rvs()

    # Modify the argument for scipy consistency
    if isinstance(ef_dist, ef.DirichletDistribution):
        scipy_x = scipy_x[0]

    actual = session.run(ef_dist.log_proba(ef_x))
    if hasattr(scipy_dist, 'logpdf'):
        desired = scipy_dist.logpdf(scipy_x)
    elif hasattr(scipy_dist, 'logpmf'):
        desired = scipy_dist.logpmf(scipy_x)
    else:
        raise NotImplementedError("%s does not support 'log*'" % scipy_dist)
    np.testing.assert_allclose(
        actual, desired, 1e-5, err_msg="inconsistent log probability for '%s': expected %s but got "
        " %s" % (ef_dist.to_str(session), desired, actual)
    )


def test_normal_linear_log_likelihood(session):
    # Generate some data
    x = np.random.normal(0, 1, (100, 3))
    theta = np.random.normal(0, 1, 3)
    predictor = np.dot(x, theta)
    tau = np.random.gamma(1)
    scale = 1 / np.sqrt(tau)
    y = np.random.normal(0, scale, 100)
    # Compare the log-likelihoods (this does NOT test the correctness for distributions but only
    # for fixed values)
    desired = scipy.stats.norm.logpdf(y, predictor, scale)
    actual = session.run(ef.NormalDistribution.linear_log_likelihood(x, y, theta, tau))
    np.testing.assert_allclose(actual, desired, 1e-5)


def test_sample_shape(session, distribution_pair):
    ef_dist, _ = distribution_pair
    assert session.run(ef_dist.sample_shape).size == ef_dist.sample_rank


def test_shape(session, distribution_pair):
    ef_dist, _ = distribution_pair
    mean, shape = session.run([ef_dist.mean, ef_dist.shape])
    np.testing.assert_equal(shape, mean.shape)


def test_batch_shape(session, distribution_pair):
    ef_dist, _ = distribution_pair
    mean, shape = session.run([ef_dist.mean, ef_dist.batch_shape])
    desired = mean.shape[:-ef_dist.sample_rank] if ef_dist.sample_rank else mean.shape
    np.testing.assert_equal(shape, desired)
