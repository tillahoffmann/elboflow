import pytest
import scipy.special
import numpy as np

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
