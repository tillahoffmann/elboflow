import uuid
import pytest
import scipy.special
import numpy as np
import tensorflow as tf

import elboflow as ef


@pytest.mark.parametrize('x, p', [
    (5, 3),
    (12, 2),
    (3.5, 1),
])
def test_lmultigamma(session, x, p):
    op = ef.lmultigamma(x, p)
    actual = session.run(op)
    desired = scipy.special.multigammaln(x, p)
    np.testing.assert_allclose(actual, desired)


def test_symmetric_logdet(session):
    x = np.random.normal(0, 1, (10, 10))
    x = np.dot(x, x.T)
    actual = session.run(ef.symmetric_log_det(x))
    desired = np.log(np.linalg.det(x))
    np.testing.assert_allclose(actual, desired, 1e-5)


@pytest.mark.parametrize('shape', [tuple(), 5, (8, 7)])
def test_get_positive_variable(session, shape):
    x = ef.get_positive_variable('x%s' % uuid.uuid4().hex, shape)
    session.run(tf.global_variables_initializer())
    np.testing.assert_array_less(0, session.run(x), "variable is not positive")


@pytest.mark.parametrize('shape', [(5, 5), (7, 5, 5)])
def test_get_tril_variable(session, shape):
    x = ef.get_tril_variable('x%s' % uuid.uuid4().hex, shape)
    session.run(tf.global_variables_initializer())
    y = session.run(x)
    np.testing.assert_array_equal(0, np.triu(y, 1), "upper triangular matrix is not zero")


@pytest.mark.parametrize('shape', [(5, 5), (7, 5, 5)])
def test_get_cholesky_variable(session, shape):
    x = ef.get_cholesky_variable('x%s' % uuid.uuid4().hex, shape)
    session.run(tf.global_variables_initializer())
    y = session.run(x)
    np.testing.assert_array_equal(0, np.triu(y, 1), "upper triangular matrix is not zero")
    i = np.arange(shape[-1])
    np.testing.assert_array_less(0, y[..., i, i], "diagonal is not positive")


@pytest.mark.parametrize('shape', [(5, 5), (7, 5, 5)])
def test_get_positive_definite_variable(session, shape):
    x = ef.get_positive_definite_variable('x%s' % uuid.uuid4().hex, shape)
    session.run(tf.global_variables_initializer())
    y = session.run(x)
    np.testing.assert_allclose(y, np.swapaxes(y, -1, -2), err_msg='matrix not symmetric')
    i = np.arange(shape[-1])
    np.testing.assert_array_less(0, y[..., i, i], "diagonal is not positive")


@pytest.mark.parametrize('shape', [(5, 5), (7, 5, 5)])
def test_get_normalized_variable(session, shape):
    x = ef.get_normalized_variable('x%s' % uuid.uuid4().hex, shape)
    session.run(tf.global_variables_initializer())
    y = session.run(x)
    np.testing.assert_allclose(np.sum(y, axis=-1), 1, 1e-5, err_msg='variables are not normalized')
