import pytest
import tensorflow as tf
import elboflow as ef


@pytest.fixture
def session():
    sess = tf.Session()
    yield sess
    sess.close()

@pytest.fixture(params=[tf.float32, tf.float64])
def dtype(request):
    _current = ef.constants.FLOATX
    ef.constants.FLOATX = request.param
    yield request.param
    ef.constants.FLOATX = _current
