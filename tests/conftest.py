import pytest
import tensorflow as tf


@pytest.fixture
def session(request):
    sess = tf.Session()
    yield sess
    sess.close()
