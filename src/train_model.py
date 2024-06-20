"""Train model and save checkpoint"""

import argparse
import catboost as cb
from catboost import Pool
import numpy as np
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from joblib import dump

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'data/proc/train.csv'
VAL_DATA = 'data/proc/val.csv'
MODEL_SAVE_PATH = 'models/Catboost_top.joblib'


def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    x_train = df_train[['rooms_count', 'author_type', 'floor', 'floors_count', 'first_floor', 'last_floor', 'total_meters', 'district']]
    y_train = df_train['price']
    df_val = pd.read_csv(VAL_DATA)
    x_val = df_val[['rooms_count', 'author_type', 'floor', 'floors_count', 'first_floor', 'last_floor', 'total_meters', 'district']]
    y_val = df_val['price']

    bustec = cb.CatBoostRegressor(iterations = 500, cat_features = ['district', 'author_type'])
    bustec.fit(x_train, np.log(y_train),
              eval_set = Pool(x_val, np.log(y_val), ['district', 'author_type']),
              cat_features=['district', 'author_type'],
               silent=True)
    dump(bustec, args.model)
    logger.info(f'Saved to {args.model}')

    y_pred = np.exp(bustec.predict(x_val))
    mae = mean_absolute_error(y_pred, y_val)

    logger.info(f'MAE = {mae:.0f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)