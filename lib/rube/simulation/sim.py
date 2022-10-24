import logging
import time

import jax
import numpy as np
from jax import numpy as jnp

from rube.simulation import met_hast
from rube.utils.prefit import load_canonical_prefit_model


class Simulation:
    def __init__(self, n_products=None, prefit_dir=None, max_q=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.initial_jaxkey = jax.random.PRNGKey(seed or 3)
        self.jaxkey = self.initial_jaxkey
        model = load_canonical_prefit_model(model_dir=prefit_dir, n_products=n_products)
        self.max_q = max_q
        self.n_products = n_products or model.stock_vocab_size
        self.n_periods = model.n_periods
        self.embedding_dim = model.embedding_dim
        self.user_vocab_size = model.user_vocab_size
        self.source_model = model
        self.period_token = 0
        self.all_users = jnp.arange(model.user_vocab_size, dtype=np.int32)
        self.price_vector_ = None
        self.data_ = None

    def set_price_vector(self, prices):
        if self.price_vector_ is None:
            # this object deals with a constant price vector:
            assert prices.shape[0] == 1
        else:
            assert prices.shape == self.price_vector_.shape
        self.price_vector_ = prices

    def get_price_vector(self):
        assert self.price_vector_ is not None
        return self.price_vector_

    def set_data(self, data):
        assert self.data_ is None
        self.data_ = data

    def get_data(self):
        assert self.data_ is not None
        return self.data_

    def set_jaxkey(self, key):
        self.jaxkey = key

    def get_jaxkey(self):
        return self.jaxkey

    def save_simdef(self, directory):
        raise NotImplementedError

    def save_data(self, directory):
        '''
        Save the sim dataset and Simulation alongside it
        '''
        # TODO: can this please be nicely compressed?
        raise NotImplementedError

    def build_data_arrays(self, bundles, us):
        """
        Prepare the simulation's results as an input for a Generator() object
        """
        ps = self.get_price_vector()
        ts = jnp.repeat(self.period_token, bundles.shape[0], axis=0)
        assert us.shape == ts.shape
        return {'q': bundles, 'p': ps, 't': ts, 'u': us}

    def build_bundles(self, n_samples, basket_size, sample_freq=50, burnin=2500, num_streams=8):
        logging.info(f'Generating starting bundles of size {basket_size}')
        seed_baskets = jnp.empty((num_streams, self.n_products), dtype=np.int8)
        jaxkeys, new_key = self.generate_keys(num_streams)
        self.set_jaxkey(new_key)
        for i in range(num_streams):
            b = met_hast.build_seed_basket(self.max_q, jaxkeys[i], self.n_products, basket_size)
            seed_baskets = seed_baskets.at[i].set(b)
        logging.info(f'Generating {n_samples} bundles with Metropolis-Hastings algorithm...')
        st = time.time()

        sim_users = jax.random.choice(new_key, self.all_users, (num_streams, 1), replace=num_streams > len(self.all_users))
        logging.info(f'These bundles belong to the {len(sim_users)} users (maybe with repeats): '
                     f'{", ".join(str(y) for y in sorted(x[0] for x in sim_users))}.')

        results = met_hast.generate_draws(self.source_model.params,
                                          self.max_q,
                                          self.jaxkey,
                                          sim_users,
                                          seed_baskets,
                                          self.get_price_vector(),
                                          jnp.array([self.period_token], dtype=jnp.int32),  # just the one time-period,
                                          n_samples=n_samples // num_streams,
                                          sample_freq=sample_freq,
                                          min_iters=burnin)
        logging.info(f'Generated {n_samples // num_streams * num_streams} baskets '
                     f'in {np.round((time.time() - st) / 60, 2)} minutes.')
        return results

    def generate_keys(self, num_streams):
        jaxkeys = jax.random.split(self.jaxkey, num=num_streams+1)
        return jaxkeys[:-1], jaxkeys[-1]


def load_simdef(directory):
    '''
    Load sufficient paramters from json to uniquely define a Simulation class object
    return a simdef object
    '''
    raise NotImplementedError


def load_sim_dataset(directory):
    '''
    Load a Simulation and sim dataset and check that they are coherent

    call simdef.init()
    return a sim dataset
    '''
    raise NotImplementedError
