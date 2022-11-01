import argparse

import numpy as np
import xgboost as xgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def main():
    args = parse_args()

    test_features = np.load(f'output/test_features_core_xgb_{args.city}.npy')

    bst = xgb.Booster()
    bst.load_model(f'output/xgb_model_core_{args.city}.json')

    dtest = xgb.DMatrix(test_features)

    if args.city == 'london':
        iteration = 11000
    elif args.city == 'madrid':
        iteration = 10000
    elif args.city == 'melbourne':
        iteration = 5000
    else:
        raise ValueError()

    test_predictions = bst.predict(data=dtest, iteration_range=(0, iteration))
    np.save(f'output/test_predictions_core_xgb_{args.city}.npy', test_predictions)


if __name__ == '__main__':
    main()
