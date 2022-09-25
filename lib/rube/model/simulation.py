import jax
import numpy as np
from jax import numpy as jnp

import rube.model
from rube.data.generator import build_signal_set


def build_seed_basket(cleaner, key, basket_size=6, n_products=None, period_token=0):
    # by setting period_token to zero, we simulate in the UNK (average) period of the data
    df = cleaner.raw_data
    user_token = jax.random.choice(key, df['user_token'].unique().shape[0])[jnp.newaxis]
    vocab_size = n_products or len(cleaner.stock_vocab)
    basket = jnp.zeros(vocab_size, dtype=np.int8)
    basket_tokens = jax.random.choice(key, np.arange(1, n_products), (basket_size,), replace=False)
    basket = basket.at[basket_tokens].set(jax.random.choice(key, jnp.arange(1, cleaner.max_seen_q), (basket_size,)))
    prices = np.zeros((1, len(cleaner.stock_vocab)), dtype=np.float32)

    prices[:, cleaner.data['product_token']] = cleaner.data['MeanPrice']

    if n_products:
        prices = prices[:, :n_products]
    return user_token, basket, prices, jnp.array([period_token])


@jax.tree_util.Partial(jax.jit, static_argnums=(6,))
def propose_new(user_token, basket, prices, period, raw_params, keys, max_q):
    """
    :param user_token:
    :param basket: we are interested in proceeding from this basket to another, following a Markov chain
    which settles down into an ergodic distribution equal to the true distribution of baskets.
    :param prices: the prices of the goods which are in force, a jnp array.
    :param raw_params: model parameters
    :param keys: two jaxkeys
    :param max_q: biggest permitted quantity that can be drawn
    :return: (the next basket, its utility, indicator variable for whether it differs from its predecessor).
    """
    key0, key1, key2, key3 = keys
    choices = build_signal_set(basket, (key0, key1, key2), max_q, 1, replace=False)
    utilities = rube.model.model.qua_model(raw_params, choices, prices, period, user_token)
    ratio = jnp.exp(utilities[1] - utilities[0])
    rand = jax.random.uniform(key3)
    idx = jnp.int32(rand < ratio)[0]  # trivially true if ratio > 1, of course
    return choices[idx], utilities[idx], idx


def generate_draws(params, max_q, draw_key, u, bs, p, t, n_samples=5000, min_iters=2500, sample_freq=50):
    """
    :param params: model parameters
    :param max_q: biggest permitted quantity that can be drawn
    :param draw_key: jaxkey
    :param u: user token. So far, this has been constant and equal to zero in applications.
    :param bs: we are interested in proceeding from these baskets (of various sizes) to another, following a Markov chain
    which settles down into an ergodic distribution equal to the true distribution of baskets.
    :param p: the prices of the goods which are in force, a jnp array.
    :param t: a period token
    :param n_samples: number of baskets to simulate
    :param min_iters: it is deemed that the ergodic distribution has not been reached until after this many iterations.
    :param sample_freq:
    :returns: a set of n_samples baskets of goods, generated with the MH algorithm
    """
    params = params.copy()
    params['A_'] = params['A_'][:bs[0].shape[0]]
    assert (rube.model.model.load_params(params)['A'][0] == 0).all()

    baskets, uts, idxs = jax.vmap(lambda b: scan_draws(b, draw_key, params, min_iters, sample_freq, n_samples, max_q, u, p, t))(bs)

    n_threads, vocab_size = bs.shape
    merged_data = baskets.reshape(n_threads * n_samples, vocab_size)

    ps = jnp.repeat(p, merged_data.shape[0], axis=0)
    ts = jnp.repeat(t, merged_data.shape[0], axis=0)
    us = jnp.repeat(u, merged_data.shape[0], axis=0)

    return {'q': merged_data, 'p': ps, 't': ts, 'u': us}


@jax.tree_util.Partial(jax.jit, static_argnums=(1))
def next_step(params, max_q, u, prices, period, carry, x):
    basket, draw_key = carry
    draw_key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(draw_key, num=5)
    keys = (subkey1, subkey2, subkey3, subkey4)
    basket, ut, idx = propose_new(u, basket, prices, period, params, keys, max_q)
    return (basket, draw_key), (basket, ut, idx)


@jax.tree_util.Partial(jax.jit, static_argnums=(0, 2))
def next_draw(sample_freq, params, max_q, u, prices, period, carry, x):
    scan_fn = jax.tree_util.Partial(next_step, params, max_q, u, prices, period)
    (end_basket, end_key), (baskets, uts, idxs) = jax.lax.scan(scan_fn, carry, None, length=sample_freq)
    return (end_basket, end_key), (end_basket, uts[-1], idxs[-1])


@jax.tree_util.Partial(jax.jit, static_argnums=(3, 4, 5, 6))
def scan_draws(init_basket, init_key, params, min_iter, sample_freq, n_samples, max_q, u, prices, period):
    init_scan_fn = jax.tree_util.Partial(next_step, params, max_q, u, prices, period) # Draw every sample for burn-in
    scan_fn = jax.tree_util.Partial(next_draw, sample_freq, params, max_q, u, prices, period) # Draw every sample_freq basket for actual simulation
    init_carry = (init_basket, init_key)
    # Burn-in loop
    carry, (_, _, _) = jax.lax.scan(init_scan_fn, init_carry, None, length=min_iter)
    # Actual draws
    _, (baskets, uts, idxs) = jax.lax.scan(scan_fn, carry, None, length=n_samples)
    return baskets, uts, idxs