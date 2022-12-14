import argparse

import numpy as np
import lightgbm as lgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def main():
    args = parse_args()

    test_features = np.load(f'output/test_features_extended_lgb_{args.city}.npy')

    bst = lgb.Booster(model_file=f'output/lgb_model_extended_{args.city}.txt')

    test_predictions = bst.predict(test_features, num_iteration=bst.best_iteration)

    if args.city == 'london' or args.city == 'melbourne':
        test_features = np.load(f'output/test_features_extended_lgb_{args.city}_v2.npy')
        bst = lgb.Booster(model_file=f'output/lgb_model_extended_{args.city}_v2.txt')
        test_predictions += bst.predict(test_features, num_iteration=100000 if args.city == 'london' else bst.best_iteration)
        test_predictions *= 0.5

    np.save(f'output/test_predictions_extended_lgb_{args.city}.npy', test_predictions)


if __name__ == '__main__':
    main()
