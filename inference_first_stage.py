import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import trange

from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def main():
    args = parse_args()

    node_id_dict = read_pickle(f'cache/road_graph/{args.city}/node_id_dict.pkl')

    counters_test = pd.read_parquet(f'data/test/{args.city}/input/counters_test.parquet')

    counters_test['node_id'] = counters_test['node_id'].apply(node_id_dict.get)

    features = []
    for test_idx in trange(100):
        df = counters_test.query(f'test_idx == {test_idx}')
        counters = [np.array([np.nan] * 4, dtype=np.float32)] * len(node_id_dict)
        for _, row in df.iterrows():
            counters[row['node_id']] = row['volumes_1h']
        counters = np.stack(counters, axis=0).reshape(-1)
        features.append(counters)

    features = np.stack(features, axis=0)

    bst = xgb.Booster()
    bst.load_model(f'output/xgb_model_{args.city}_month.json')
    month_predictions = bst.predict(xgb.DMatrix(features))
    gbm = lgb.Booster(model_file=f'output/lgb_model_{args.city}_month.txt')
    month_predictions += gbm.predict(features)
    month_predictions *= 0.5
    month_predictions = month_predictions.round().astype(np.uint8)

    bst = xgb.Booster()
    bst.load_model(f'output/xgb_model_{args.city}_day_of_week.json')
    day_of_week_predictions = bst.predict(xgb.DMatrix(features))
    gbm = lgb.Booster(model_file=f'output/lgb_model_{args.city}_day_of_week.txt')
    day_of_week_predictions += gbm.predict(features)
    day_of_week_predictions *= 0.5
    day_of_week_predictions = day_of_week_predictions.round().astype(np.uint8)

    bst = xgb.Booster()
    bst.load_model(f'output/xgb_model_{args.city}_t.json')
    t_predictions = bst.predict(xgb.DMatrix(features))
    gbm = lgb.Booster(model_file=f'output/lgb_model_{args.city}_t.txt')
    t_predictions += gbm.predict(features)
    t_predictions *= 0.5
    t_predictions = t_predictions.round().astype(np.uint8)

    np.save(f'output/{args.city}_month.npy', month_predictions)
    np.save(f'output/{args.city}_day_of_week.npy', day_of_week_predictions)
    np.save(f'output/{args.city}_t.npy', t_predictions)


if __name__ == '__main__':
    main()
