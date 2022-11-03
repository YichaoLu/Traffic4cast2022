import argparse

import numpy as np
import xgboost as xgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def main():
    args = parse_args()

    train_features = np.load(f'output/train_features_core_{args.city}.npy')
    train_targets = np.load(f'output/train_targets_core_{args.city}.npy')
    train_weights = np.load(f'output/train_weights_core_{args.city}.npy')
    valid_features = np.load(f'output/valid_features_core_{args.city}.npy')
    valid_targets = np.load(f'output/valid_targets_core_{args.city}.npy')
    valid_weights = np.load(f'output/valid_weights_core_{args.city}.npy')

    dtrain = xgb.DMatrix(data=train_features, label=train_targets, weight=train_weights)
    dvalid = xgb.DMatrix(data=valid_features, label=valid_targets, weight=valid_weights)

    params = {
        'booster': 'gbtree',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'tree_method': 'gpu_hist',
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'seed': 0
    }

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100_000_000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=1000
    )
    bst.save_model(f'output/xgb_model_core_{args.city}.json')


if __name__ == '__main__':
    main()
