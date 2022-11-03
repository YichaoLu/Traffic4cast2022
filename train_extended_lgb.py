import argparse

import lightgbm as lgb
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def main():
    args = parse_args()

    train_features = np.load(f'output/train_features_extended_{args.city}.npy')
    train_targets = np.load(f'output/train_targets_extended_{args.city}.npy')
    valid_features = np.load(f'output/valid_features_extended_{args.city}.npy')
    valid_targets = np.load(f'output/valid_targets_extended_{args.city}.npy')

    lgb_train = lgb.Dataset(data=train_features, label=train_targets, categorical_feature=[0, 1, 2])
    lgb_valid = lgb.Dataset(data=valid_features, label=valid_targets, categorical_feature=[0, 1, 2])

    params = {
        'boosting': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'l1',
        'learning_rate': 0.1,
        'num_leaves': 2 ** 5,
        'max_depth': 5,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
        'device_type': 'cpu',
        'num_threads': 0,
        'seed': 0,
        'verbosity': 0
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=500_000_000,
        valid_sets=[lgb_train, lgb_valid],
        categorical_feature=[0, 1, 2],
        early_stopping_rounds=1_000
    )
    gbm.save_model(f'output/lgb_model_extended_{args.city}.txt')


if __name__ == '__main__':
    main()
