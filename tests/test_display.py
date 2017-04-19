import pytest
import numpy as np
import elboflow as ef


@pytest.fixture(params=['univariate', 1, 3])
def distribution(request):
    if request.param == 'univariate':
        return ef.NormalDistribution(np.random.normal(), np.random.gamma(1))
    else:
        return ef.MultiNormalDistribution(np.random.normal(size=request.param),
                                          np.diag(np.random.gamma(1, size=request.param)))

def test_plot_proba(session, distribution):
    # TODO: include univariate distributions with non-empty batch shape
    if distribution.sample_rank != 0:
        pytest.skip("plot_proba is only supported for scalar distributions")
    ef.plot_proba(session, distribution)


def test_plot_cov(session, distribution):
    if isinstance(distribution, ef.NormalDistribution):
        pytest.skip()
    ef.plot_cov(session, distribution)


def test_plot_comparison(session, distribution):
    ef.plot_comparison(session, distribution, session.run(distribution.mean))
