#!/usr/bin/env python

import os
import warnings
import logging
import argparse
import pathlib
import time
import json
from rube.data.uci import UCIGenerator
from rube.model.model import RubeJaxModel

STANDARD_FIT_DIR = pathlib.Path().absolute().parent.parent / 'RUBE_fits'

warnings.simplefilter("ignore")

# create logger
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, force=True)

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=12, help='Number of dimensions')
parser.add_argument('--stock_vocab_size', type=int, default=7500, help='Max number of products to encode')
parser.add_argument('--user_vocab_size', type=int, default=1250, help='Max number of users to encode')
parser.add_argument('--ns', type=int, default=99, help='Number of negative samples')
parser.add_argument('--mb', type=int, default=1024, help='Minibatch size')
parser.add_argument('--n_epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--max_quantity', type=int, default=6,
                    help='Max quantity to use in generated fake data and in truncated real data')
parser.add_argument('--dataset_lines', type=int, default=2000000, help='Number of lines of the dataset to ingest')
parser.add_argument('--begin_week', type=int, default=0, help='First week number of included data')
parser.add_argument('--end_week', type=int, default=None, help='Last week number of included data')
parser.add_argument('--min_visits', type=int, default=0,
                    help='minimum number of times a customer must show up '
                         'in the dataset to be included in cleaned data')
parser.add_argument('--min_baskets', type=int, default=0,
                    help='minimum number of baskets a product must show up in '
                         'in order to be included in the cleaned data')
parser.add_argument('--min_average_spend', type=int, default=0,
                    help='minimum average spend of a customer to be included in the cleaned data')
parser.add_argument('--fit_dir', type=str, default=None, help='Directory in which to perform the fit')
parser.add_argument("--dataset", choices=['uci'], default='uci', help="pick dataset")
parser.add_argument('--repeat_holdout', type=int, default=1,
                    help="Generate this many signal sets for every held out basket")
parser.add_argument('--holdout_size', type=float, default=1024, help="Proportion of data to hold out")
parser.add_argument('--step_size', type=float, default=0.01, help="Step size (for Adam optimizer)")
parser.add_argument('--period_in_weeks', type=int, default=5200,
                    help="The data can be divided into periods. This is the time in weeks of each period. "
                         "The default is set to 100 years, so that there is only one period.")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--delete_unks', default=False, action='store_true',
                    help='proceed as if purchases of goods that are too rare to track (of UNKS), never happened.')
parser.add_argument('--keep_singleton_baskets', default=False, action='store_true',
                    help='When argument is present, do not delete singleton baskets')
args = parser.parse_args()

# code to reconfigure args (currently it does only one thing)
args.remove_singleton_baskets = not args.keep_singleton_baskets
kwargs = vars(args)
for key in ['keep_singleton_baskets']:
    kwargs.pop(key, None)


def get_data_generator(dataset_string):
    if dataset_string == 'uci':
        logging.info(f"Using the 'UCI archive, online_retail_II.xlsx' dataset.\n")
        return UCIGenerator
    else:
        raise NotImplementedError

    def prepare_fitting_directory(option_args):
        fits_dir = option_args.fit_dir or STANDARD_FIT_DIR
        fdir = pathlib.Path(fits_dir) / time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(fdir)
        with open(fdir / 'parameters.json', 'w+') as param_file:
            json.dump(vars(option_args), param_file)
        logging.info(f"The fitting directory is at {fdir}.")
        return fdir

    if args.delete_unks:
        raise NotImplementedError

# create logger
warnings.simplefilter("ignore")
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, force=True)

fit_dir = prepare_fitting_directory(args)
gen = get_data_generator(args.dataset)

dg = gen(stock_vocab_size=args.stock_vocab_size,
         user_vocab_size=args.user_vocab_size,
         batch_size=args.mb,
         neg_samples=args.ns,
         period_in_weeks=args.period_in_weeks,
         seed=args.seed,
         test_size=args.holdout_size,
         repeat_holdout=args.repeat_holdout,
         begin_week=args.begin_week,
         end_week=args.end_week,
         n_lines=args.dataset_lines,
         min_visits=args.min_visits,
         min_baskets=args.min_baskets,
         min_average_spend=args.min_average_spend,
         max_accepted_quantity=args.max_quantity,
         remove_singleton_baskets=args.remove_singleton_baskets)

model = RubeJaxModel(stock_vocab_size=dg.get_stock_vocab_size(),
                     n_periods=dg.get_n_periods(),
                     user_vocab_size=dg.get_user_vocab_size(),
                     embedding_dim=args.K,
                     seed=args.seed,
                     step_size=args.step_size)

model.training_loop(dg, epochs=args.n_epochs, fit_dir=fit_dir)


if __name__ == "__main__":
    main()
