import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split

import logging


class Generator:
    def __init__(self, data, batch_size, neg_samples, max_quantity, stock_vocab, n_periods, replace=True,
                 user_vocab_size=None,
                 repeat_holdout=1, seed=None, shuffle=False, test_size=0.02):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        else:
            self.key = jax.random.PRNGKey(np.random.randint(10000000))
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.max_quantity = max_quantity
        self.repeat_holdout = repeat_holdout
        self.replace = replace
        self.test_size = test_size if test_size < 1 else int(test_size)
        effective_test_size = self.calc_effective_test_size(data['q'].shape[0])
        if effective_test_size > self.batch_size:
            logging.warning(f'Effective test size ({effective_test_size}) is bigger than the batch '
                            f'size ({self.batch_size}), as a result it is possible that memory issues will arise, '
                            f'even if the training would otherwise be possible with these parameters.')
        logging.info(f"A proportion {test_size} of the data will be reserved as holdout.")
        if replace:
            logging.warning("The Signal Sets' -ve items are generated with replacement. "
                            "While this is a speedup, it may fail our assumptions for consistency.")
        self.stock_vocab = stock_vocab
        if user_vocab_size is not None:
            self.user_vocab_size = user_vocab_size
            self.contains_user_data = True
        else:
            self.contains_user_data = False
        self.training_data, self.holdout = self.arrange_data(data)
        self.n_samples = self.training_data['q'].shape[0]
        self.n_periods = n_periods
        self._index = 0

    def calc_effective_test_size(self, nrows):
        if self.test_size >= 1:
            return self.test_size * self.repeat_holdout
        else:
            return nrows * self.test_size

    def get_stock_vocab_size(self):
        return len(self.stock_vocab)

    def get_n_periods(self):
        return self.n_periods

    def get_user_vocab_size(self):
        if self.contains_user_data:
            return self.user_vocab_size
        else:
            raise ValueError("Generator doesn\'t contain user data")

    def get_n_iter(self):
        return np.int32(np.ceil(self.n_samples / self.batch_size))

    def arrange_data(self, x, sort_users=False):
        """
        @param x: the training dataset in 'concise' dictionary form
        @param sort_users: (before batching) sort the training data by user id
        """
        train = dict()
        test = dict()

        # build indices for test and train:
        index_of_datapoints = np.arange(len(x['q']))
        tr, te = train_test_split(index_of_datapoints, random_state=None, test_size=self.test_size)

        if 'u' in x:  # the usual case: there is user_token data
            if sort_users:
                training_users = x['u'][tr].T
                sort,  = np.argsort(training_users)
                logging.info("Sorting the training data by user id before fitting to it")
                tr = tr[sort]
            train['u'], test['u'] = x['u'][tr], x['u'][te]
        train['q'], test['q'] = x['q'][tr], x['q'][te]
        train['t'], test['t'] = x['t'][tr], x['t'][te]

        n_prices, n_goods = x['p'].shape
        if n_prices > 1:  # the usual case
            assert n_prices == len(x['q'])
            train['p'], test['p'] = x['p'][tr], x['p'][te]
        else:  # prices are constant, so are in a singleton vector, e.g. from a simulation
            train['p'], test['p'] = x['p'], x['p']

        return train, self.arrange_test_data(test)

    def arrange_test_data(self, test_batch):
        """
        The holdout/test data is arranged off-line into a batch structure, building signal sets etc.
        """
        num = test_batch["q"].shape[0]
        logging.info(f'Generating {self.repeat_holdout} signal set(s) for each of {num} held-out baskets')

        rep = lambda x: np.repeat(test_batch[x], self.repeat_holdout, axis=0)

        arranged = {'quantity': self.batch_signal_set(rep('q'), randomize=True),
                    'prices': rep('p')[:, None, :],
                    'period': rep('t')}

        if 'u' in test_batch:  # the usual case: there is user_token data
            arranged['users'] = rep('u')

        return arranged

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self.n_samples:
            self._index = 0
            raise StopIteration  # end of one epoch

        rng = slice(self._index, self._index + self.batch_size)
        # the line below enables sampling from the dataset with replacement:
        # rng = jax.random.randint(self.key, (self.batch_size,), 0, self.n_samples, dtype=jnp.int16)
        data = self.training_data

        periods = data['t'][rng]
        prices = data['p'][rng] if len(data['p']) > 1 else data['p']

        out = {'quantity': self.batch_signal_set(data['q'][rng], randomize=False),
               'prices': prices[:, None, :],
               'period': periods}

        if 'u' in data:  # the usual case: there is user_token data
            out['users'] = data['u'][rng]

        self._index = self._index + self.batch_size
        return out

    def batch_signal_set(self, baskets, randomize=False):
        """
        :param baskets: a collection of baskets: our data
        :param randomize: if true, generate a new random key for every basket, if false than only a new key for every batch
        :return: a batch of signal sets in the form of a 3d array
        """
        if randomize:
            keys = jax.random.split(self.key, 3 * baskets.shape[0] + 1)
            self.key = keys[-1]
            keys = keys[:-1].reshape(baskets.shape[0], 3, 2)
            return jax.vmap(build_signal_set, in_axes=(0, 0, None, None, None))(jnp.array(baskets),
                                                                                keys,
                                                                                self.max_quantity,
                                                                                self.neg_samples,
                                                                                self.replace)
        else:
            keys = jax.random.split(self.key, 4)
            self.key = keys[-1]
            return jax.vmap(build_signal_set, in_axes=(0, None, None, None, None))(jnp.array(baskets),
                                                                                   keys[:-1],
                                                                                   self.max_quantity,
                                                                                   self.neg_samples,
                                                                                   self.replace)


@jax.jit
def normalise(x):
    return x / x.sum()


@jax.tree_util.Partial(jax.jit, static_argnums=(2, 3, 4))
def build_signal_set(basket, keys, max_quantity, neg_samples, replace):
    '''
    :param basket: an initial basket
    :param keys: 3 separate keys to use for randomization
    :param max_quantity: max quantity of items to be added to negative samples
    :param neg_samples: number of alternative baskets (negative samples) to generate
    :param replace: if True, sample with replacement, this is much faster
    :return: jnp.ndarray containing the initial basket and negative samples stacked over axis 0
    '''
    nonzero = (basket > 0)[1:].astype(jnp.int8)
    poss_neg_samples = (1 - nonzero)
    # we start `rng` at 1 so that we don't pick UNK as a positive or negative item:
    rng = jnp.arange(1, basket.shape[0])
    item_index = jax.random.choice(keys[0], rng, p=normalise(nonzero))
    fake_item_idx = jax.random.choice(keys[1], rng, (neg_samples,), replace=replace, p=normalise(poss_neg_samples))
    fake_item_quantity = jax.random.choice(keys[2], jnp.arange(1, max_quantity + 1), replace=True, shape=(neg_samples,))

    bout = jnp.repeat(basket[None, :], neg_samples + 1, axis=0)

    # clear out the positive item from the parts of bout which are to describe negative samples:
    bout = bout.at[1:, item_index].set(0)

    # assign the new negative items to fake baskets (in a vectorised fashion)
    arange = jnp.arange(1, neg_samples + 1)
    bout = bout.at[arange, fake_item_idx].set(fake_item_quantity)

    return bout
