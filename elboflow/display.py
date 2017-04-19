from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np

from .util import minmax


def evaluate_proba(session, distribution, start=None, stop=None, num=50, scale=3):
    """
    Evaluate the probability density function of a distribution.

    Parameters
    ----------
    session : tf.Session
        tensorflow session used to evaluate the PDF.
    distribution : Distribution
        distribution to evaluate.
    start : float or None
        lower limit of the interval to evaluate the PDF on. If `None`, `mean - scale * std` is used.
    stop : float or None
        upper limit of the interval to evaluate the PDF on. If `None`, `mean + scale * std` is used.
    num : int
        number of subdivisions of the interval.
    scale : float
        number of standard deviations to evaluate the PDF over. Ignored if `start` and `stop` are
        given.
    """
    assert distribution.sample_rank == 0, "only scalar distributions are supported"

    # Evaluate the limits
    if start is None or stop is None:
        mean, std = session.run([distribution.mean, distribution.std])
        start = mean - scale * std if start is None else start
        stop = mean + scale * std if stop is None else stop

    # Evaluate the probability density function
    shape = [num, *np.ones_like(session.run(distribution.shape))]
    lin = np.reshape(np.linspace(0, 1, num), shape)
    lin = start + lin * (stop - start)
    return lin, np.exp(session.run(distribution.log_proba(lin)))


def plot_proba(session, distribution, start=None, stop=None, num=50, scale=3, reference=None,
               ax=None, **kwargs):
    """
    Plot the probability density function of a (multivariate) distribution.

    Parameters
    ----------
    session : tf.Session
        tensorflow session used to evaluate the PDF.
    distribution : Distribution
        distribution to evaluate.
    start : float or None
        lower limit of the interval to evaluate the PDF on. If `None`, `mean - scale * std` is used.
    stop : float or None
        upper limit of the interval to evaluate the PDF on. If `None`, `mean + scale * std` is used.
    num : int
        number of subdivisions of the interval.
    scale : float
        number of standard deviations to evaluate the PDF over. Ignored if `start` and `stop` are
        given.
    reference : np.ndarray or None
        reference values to plot as vertical lines
    ax :
        axes to plot into
    kwargs : dict
        keyword arguments passed on to `ax.plot`
    """
    # Plot the PDF
    ax = ax or plt.gca()
    lin, pdf = evaluate_proba(session, distribution, start, stop, num, scale)
    shape = (lin.shape[0], -1)
    lines = ax.plot(np.reshape(lin, shape), np.reshape(pdf, shape), **kwargs)

    # Plot the reference values if given
    if reference is not None:
        for line, value in zip(lines, np.atleast_1d(reference)):
            ax.axvline(value, color=line.get_color())

    return lines


def plot_comparison(session, distribution, reference, scale=3, plot_diag=True, aspect='equal',
                    ax=None, **kwargs):
    """
    Plot the distribution mean and errors against reference values.

    Parameters
    ----------
    session : tf.Session
        tensorflow session used to evaluate the PDF.
    distribution : Distribution
        distribution to evaluate.
    reference : np.ndarray or None
        reference values to plot against
    scale : float
        multiplicative factor for the standard deviation.
    plot_diag : bool
        whether to plot a diagonal reference line.
    aspect : float or str
        aspect ratio.
    ax :
        axes to plot into
    kwargs : dict
        keyword arguments passed on to `ax.errorbar`
    """
    # Plot the estimates against the reference
    ax = ax or plt.gca()
    kwargs_default = {
        'linestyle': 'none',
        'marker': '.',
    }
    kwargs_default.update(kwargs)
    reference = np.atleast_1d(reference)
    y, yerr = session.run([distribution.mean, distribution.std])
    lines = ax.errorbar(reference.ravel(), y.ravel(), yerr.ravel() * scale, **kwargs_default)

    # Plot a diagonal line
    if plot_diag:
        xy = minmax(reference)
        ax.plot(xy, xy, linestyle=':', color=lines[0].get_color())

    # Set the aspect ratio
    if aspect:
        ax.set_aspect(aspect)

    return lines


def plot_cov(session, distribution, vmin='auto', vmax='auto', cmap='coolwarm', ax=None, **kwargs):
    """
    Plot the covariance matrix of a multivariate distribution.

    Parameters
    ----------
    session : tf.Session
        tensorflow session used to evaluate the PDF.
    distribution : Distribution
        distribution to evaluate.
    vmin : float, str, or None
        minimum value for the color map.
    vmax : float, str, or None
        maximum value for the color map.
    cmap : str
        color map.
    ax :
        axes to plot into
    kwargs : dict
        keyword arguments passed on to `ax.imshow`
    """
    ax = ax or plt.gca()
    cov = session.run(distribution.cov)
    if vmax == 'auto':
        vmax = np.max(np.abs(cov))
    if vmin == 'auto':
        vmin = - np.max(np.abs(cov))

    return ax.imshow(cov, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)


def ellipse_from_cov(xy, cov, scale=3, **kwargs):
    """
    Create an ellipse from a covariance matrix.

    Parameters
    ----------
    xy : np.ndarray
        position of the ellipse
    cov : np.ndarray
        covariance matrix
    scale : float
        scale of the ellipse (default is three standard deviations)
    kwargs : dict
        keyword arguments passed on to `matplotlib.patches.Ellipse`

    Returns
    -------
    ellipse
    """
    evals, evecs = np.linalg.eigh(cov)
    angle = np.arctan2(*evecs[:, 1])
    width, height = scale * np.sqrt(evals)
    return mpatches.Ellipse(xy, width, height, np.rad2deg(angle), **kwargs)
