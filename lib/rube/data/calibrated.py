import numpy as np

from rube.data.generator import Generator


class CalibratedDataGenerator(Generator):
    """
    This draws baskets from a Metropolis Hastings algorithm based on a previously fitted model.
    This facilitates a test of our fitting whereby we refit to the output of a known model based on known parameters.
    The Metropolis Hastings algorithm borrows from our Signal Set technology.
    """
    def __init__(self, sim, batch_size, neg_samples, seed=None):

        super(CalibratedDataGenerator, self).__init__(sim.get_data(), batch_size, neg_samples,
                                                      sim.max_q,
                                                      stock_vocab=np.arange(sim.n_products),
                                                      n_periods=sim.n_periods,
                                                      seed=seed)
