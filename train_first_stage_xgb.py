import argparse
import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def main():
    args = parse_args()

    node_id_dict = read_pickle(f'cache/road_graph/{args.city}/node_id_dict.pkl')

    counters = pd.read_parquet(f'data/loop_counter/{args.city}/counters_daily_by_node.parquet')

    counters['node_id'] = counters['node_id'].apply(node_id_dict.get)

    counters = counters.sample(frac=1.0)

    train_size = int(len(counters) * 0.9)
    train_counters, valid_counters = counters.iloc[: train_size], counters.iloc[train_size:]

    train_features, train_targets_month, train_targets_day_of_week, train_targets_t = [], [], [], []
    for day in train_counters['day'].unique():
        df = train_counters.query(f'day == "{day}"')
        date = datetime.datetime.strptime(day, '%Y-%m-%d').date()
        day_of_week = date.weekday()
        month = date.month
        for t in range(4, 96):
            features = [np.array([np.nan] * 4, dtype=np.float32)] * len(node_id_dict)
            for _, row in df.iterrows():
                features[row['node_id']] = row['volume'][t - 4: t]
            features = np.stack(features, axis=0).reshape(-1)
            train_features.append(features)
            train_targets_month.append(month)
            train_targets_day_of_week.append(day_of_week)
            train_targets_t.append(t)
    train_features = np.array(train_features, dtype=np.float32)
    train_targets_month = np.array(train_targets_month, dtype=np.float32)
    train_targets_day_of_week = np.array(train_targets_day_of_week, dtype=np.float32)
    train_targets_t = np.array(train_targets_t, dtype=np.float32)

    valid_features, valid_targets_month, valid_targets_day_of_week, valid_targets_t = [], [], [], []
    for day in valid_counters['day'].unique():
        df = valid_counters.query(f'day == "{day}"')
        date = datetime.datetime.strptime(day, '%Y-%m-%d').date()
        day_of_week = date.weekday()
        month = date.month
        for t in range(4, 96):
            features = [np.array([np.nan] * 4, dtype=np.float32)] * len(node_id_dict)
            for _, row in df.iterrows():
                features[row['node_id']] = row['volume'][t - 4: t]
            features = np.stack(features, axis=0).reshape(-1)
            valid_features.append(features)
            valid_targets_month.append(month)
            valid_targets_day_of_week.append(day_of_week)
            valid_targets_t.append(t)
    valid_features = np.array(valid_features, dtype=np.float32)
    valid_targets_month = np.array(valid_targets_month, dtype=np.float32)
    valid_targets_day_of_week = np.array(valid_targets_day_of_week, dtype=np.float32)
    valid_targets_t = np.array(valid_targets_t, dtype=np.float32)

    params = {
        'booster': 'gbtree',
        'eta': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'tree_method': 'approx',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 1995
    }

    dtrain = xgb.DMatrix(data=train_features, label=train_targets_month)
    dvalid = xgb.DMatrix(data=valid_features, label=valid_targets_month)
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100_000_000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=10_000
    )
    bst.save_model(f'output/xgb_model_{args.city}_month.json')

    dtrain = xgb.DMatrix(data=train_features, label=train_targets_day_of_week)
    dvalid = xgb.DMatrix(data=valid_features, label=valid_targets_day_of_week)
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100_000_000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=10_000
    )
    bst.save_model(f'output/xgb_model_{args.city}_day_of_week.json')

    dtrain = xgb.DMatrix(data=train_features, label=train_targets_t)
    dvalid = xgb.DMatrix(data=valid_features, label=valid_targets_t)
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100_000_000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=10_000
    )
    bst.save_model(f'output/xgb_model_{args.city}_t.json')


if __name__ == '__main__':
    main()
