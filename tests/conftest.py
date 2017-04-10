import pytest
import tensorflow as tf


@pytest.fixture
def session():
    sess = tf.Session()
    yield sess
    sess.close()


@pytest.fixture(params=[True, False], ids=['reduce', 'pointwise'])
def reduce(request):
    return request.param
