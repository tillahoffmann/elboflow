import pytest
import elboflow as ef
import tensorflow as tf
import numpy as np


@pytest.fixture
def model():
    def evaluate_log_joint(x):
        # Define the mean that we want to infer
        mu = ef.NormalDistribution(
            ef.get_variable("mu_mean", []),
            ef.get_positive_variable("mu_precision", [])
        )
        # Define a prior
        mu_prior = ef.NormalDistribution(0, 1e-3)
        # Return the log joint and the factor
        return tf.reduce_sum(mu.log_proba(x)) + mu_prior.log_proba(mu), {'mu': mu}

    x = np.random.normal(0, 1, 100)
    return ef.Model(evaluate_log_joint, [x])


def test_optimize(model):
    # Optimize the model
    trace = model.optimize(1000, [model.loss])
    loss = trace[model.loss]
    assert loss[0] > loss[-1], "expected loss to decrease"
    # Make sure the result is sensible
    mu = model["mu"]
    mean, std = model.run([mu.mean, mu.std])
    assert np.abs(mean / std) < 3, "expected 0 but got %s +- %s" % (mean, std)


def test_run(model):
    elbo = model.run(model.elbo, {model['mu']: 0.0})
    assert np.isfinite(elbo), "ELBO is not finite using a `feed_dict` containing distributions"
