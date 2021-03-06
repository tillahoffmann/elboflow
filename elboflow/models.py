import numbers
import functools as ft
import numpy as np
import tensorflow as tf

from .display import *
from .util import *


class Model:
    """
    Base class for variational Bayesian models.

    Parameters
    ----------
    setup : callable
        function to evaluate the expected log joint distribution and the factors approximating the
        posterior.
    args : list
        arguments passed to `setup`
    create_optimizer : callable
        function to create an optimizer.
    create_session : callable
        function to create a session.
    debug : bool
        whether to return the dictionary of locals from the `setup` function.
    """
    def __init__(self, *args, setup=None, create_optimizer=None, create_session=None, debug=False):
        if create_optimizer is not None:
            self.create_optimizer = create_optimizer
        if create_session is not None:
            self.create_session = create_session
        if setup is not None:
            self.setup = setup
        self.args = args

        with tf.Graph().as_default() as self.graph:
            self.log_likelihood, self.factors, self.priors, self.debug_locals = \
                    self.setup(*self.args, debug=debug)
            # Evaluate the entropies
            self.entropies = {key: value.entropy for key, value in self.factors.items()}
            # Compute the lower bound
            self.elbo = self.log_likelihood + \
                sum([tf.reduce_sum(entropy) for entropy in self.entropies.values()]) + \
                sum([tf.reduce_sum(prior(self.factors[key])) for key, prior in self.priors.items()])
            # Define a loss
            self.loss = - self.elbo
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = self.create_learning_rate()
            self.optimizer = self.create_optimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            self.session = self.create_session()
            self.session.run(tf.global_variables_initializer())

    def create_optimizer(self, learning_rate):  # pylint: disable=E0202
        """
        Create a tensorflow optimizer.
        """
        return tf.train.AdamOptimizer(learning_rate)

    def create_session(self):  # pylint: disable=E0202
        """
        Create a tensorflow session.
        """
        return tf.Session()

    def create_learning_rate(self):
        """
        Create a learning rate variable.
        """
        return tf.Variable(0.1, False, name='learning_rate')

    def setup(self, *args, debug=False):  # pylint: disable=E0202
        """
        Evaluate the expected value of the log joint distribution.

        Returns
        -------
        log_likelihood : tf.Tensor
            expected value of the log joint distribution.
        factors : dict[str, Distribution]
            dictionary of factors of the approximate posterior keyed by factor name.
        priors : dict[str, callable]
            dictionary of callables to evaluate the prior keyed by factor name. By convention,
            the callables should only depend on hyperparameters. Hierarchical priors should be
            included in the `log_likelihood` term.
        debug : bool
            whether to return the dictionary of locals.
        """
        raise NotImplementedError

    def optimize(self, steps, fetches=None, feed_dict=None, break_on_interrupt=True, tqdm=None):
        """
        Optimize the evidence lower bound and trace statistics.

        Parameters
        ----------
        steps : int or iterable
            number of steps or iterator
        fetches : list
            operations to trace over the optimization process
        break_on_interrupt : bool
            whether to break the optimization loop on `KeyboardInterrupt` or raise an exception.
        """
        if isinstance(steps, numbers.Integral):
            steps = range(steps)
        if tqdm is not None:
            steps = tqdm(steps)

        # Prepare the fetches
        if fetches is None:
            fetches = []
        elif isinstance(fetches, (tf.Tensor, tf.Variable, str)):
            fetches = [fetches]
        elif hasattr(fetches, '__iter__'):
            fetches = list(fetches)
        else:
            raise ValueError("cannot convert %s to valid fetches" % fetches)

        _fetches = [self.train_op] + fetches

        trace = {fetch: [] for fetch in fetches}
        try:
            # Run the optimization
            for _ in steps:
                _, *values = self.session.run(_fetches, feed_dict)
                # Add the values to the dictionary
                for fetch, value in zip(fetches, values):
                    trace[fetch].append(value)
        except KeyboardInterrupt:
            if not break_on_interrupt:
                raise

        # Convert to arrays
        return {key: np.asarray(value) for key, value in trace.items()}

    @ft.wraps(tf.Session.run)
    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        original = feed_dict or {}
        feed_dict = {}
        for key, value in original.items():
            # Get all the statistics of this distribution
            if isinstance(key, BaseDistribution):
                feed_dict.update(key.feed_dict(value))
            # Just forward the feed values transparently
            else:
                feed_dict[key] = value
        return self.session.run(fetches, feed_dict, options, run_metadata)

    def __getitem__(self, key):
        return self.factors[key]

    @ft.wraps(plot_comparison)
    def plot_comparison(self, factor, reference=None, scale=3, plot_diag=True, aspect='equal',
                        ax=None, **kwargs):
        # Use the graph in case not all operations are cached
        with self.graph.as_default():
            return plot_comparison(self.session, self.factors[factor], reference, scale,
                                   plot_diag, aspect, ax, **kwargs)

    @ft.wraps(plot_proba)
    def plot_proba(self, factor, start=None, stop=None, num=50, scale=3, reference=None,
                   ax=None, **kwargs):
        with self.graph.as_default():
            return plot_proba(self.session, self.factors[factor], start, stop, num, scale,
                              reference, ax, **kwargs)

    @ft.wraps(plot_cov)
    def plot_cov(self, factor, vmin='auto', vmax='auto', cmap='coolwarm', ax=None, **kwargs):
        with self.graph.as_default():
            return plot_cov(self.session, self.factors[factor], vmin, vmax, cmap, ax, **kwargs)
