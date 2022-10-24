from rube.data.fake import FakeDataGenerator
from rube.model.model import RubeJaxModel, load_params

import jax.numpy as jnp
import logging

BATCH_SIZE = 256
NEG_SAMPLES = 3
CONTEXT_SIZE = 3
MAX_QUANTITY = 10
N_SAMPLES = 1000
N_PERIODS = 101
STOCK_VOCAB_SIZE = 20
EMBEDDING_DIM = 8
EPOCHS = 2
USER_VOCAB_SIZE = 10
SEED = 42

GENERATOR_OPTIONS = dict(batch_size=BATCH_SIZE, neg_samples=NEG_SAMPLES, context_size=CONTEXT_SIZE, n_periods=N_PERIODS,
                         seed=SEED, max_quantity=MAX_QUANTITY, n_samples=N_SAMPLES, stock_vocab=range(STOCK_VOCAB_SIZE))

MODEL_OPTIONS = dict(stock_vocab_size=STOCK_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, n_periods=N_PERIODS, seed=SEED)


def summaries(params):
    return jnp.round(jnp.array([params['A'].sum(),
                                params['b'].sum(),
                                params['c'].sum(),
                                params['d_1'][0, 0],
                                params['d_2'][0, 0],
                                params['d_3'][0, 0]]), 5)


def test_generator():
    dg = FakeDataGenerator(**GENERATOR_OPTIONS, generate_users=False)
    x = next(dg)
    assert x['quantity'].shape == (BATCH_SIZE, NEG_SAMPLES + 1, STOCK_VOCAB_SIZE)
    assert x['prices'].shape == (BATCH_SIZE, 1, STOCK_VOCAB_SIZE)


def test_generator_with_user():
    dg = FakeDataGenerator(**GENERATOR_OPTIONS, generate_users=True, user_vocab_size=USER_VOCAB_SIZE)
    x = next(dg)
    assert x['quantity'].shape == (BATCH_SIZE, NEG_SAMPLES + 1, STOCK_VOCAB_SIZE)
    assert x['prices'].shape == (BATCH_SIZE, 1, STOCK_VOCAB_SIZE)
    assert x['users'].shape == (BATCH_SIZE,)


def test_model():
    dg = FakeDataGenerator(**GENERATOR_OPTIONS)
    model = RubeJaxModel(**MODEL_OPTIONS)
    model.training_loop(dg, epochs=EPOCHS)
    params = load_params(model.params)
    assert jnp.isclose(summaries(params), jnp.array([1.8818680e+01, -1.6399999e-03,  0.0000000e+00,
                                                     5.6281000e-01,  1.1360000e-02,  4.1520000e-01])).all()


def test_model_with_users():
    dg = FakeDataGenerator(**GENERATOR_OPTIONS, generate_users=True, user_vocab_size=USER_VOCAB_SIZE)
    model = RubeJaxModel(**MODEL_OPTIONS, user_vocab_size=USER_VOCAB_SIZE)
    model.training_loop(dg, epochs=EPOCHS)
    params = load_params(model.params)
    assert jnp.isclose(summaries(params), jnp.array([1.8933319e+01, 2.7313800e+00, 0.0000000e+00,
                                                     4.7241998e-01, 9.4900001e-03, 5.2410001e-01])).all()

