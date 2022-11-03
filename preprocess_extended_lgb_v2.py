import argparse

import networkx as nx
import numpy as np
from tqdm import tqdm

from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def eta_smoothing(eta_list):
    assert len(eta_list) == 9
    eta_sum = eta_list[0] * 1 + eta_list[1] * 16 + eta_list[2] * 81 + eta_list[3] * 256 + eta_list[4] * 625
    eta_count = 1 + 16 + 81 + 256 + 625
    if not np.isnan(eta_list[5]):
        eta_sum += eta_list[5] * 256
        eta_count += 256
    if not np.isnan(eta_list[6]):
        eta_sum += eta_list[6] * 81
        eta_count += 81
    if not np.isnan(eta_list[7]):
        eta_sum += eta_list[7] * 16
        eta_count += 16
    if not np.isnan(eta_list[8]):
        eta_sum += eta_list[8] * 1
        eta_count += 1
    return eta_sum / eta_count


def preprocess_test(args: argparse.Namespace):
    node_id_dict = read_pickle(path=f'cache/road_graph/{args.city}/node_id_dict.pkl')
    road_graph_edges = read_pickle(path=f'cache/road_graph/{args.city}/road_graph_edges.pkl')

    road_graph = nx.DiGraph()
    road_graph.add_nodes_from(range(len(node_id_dict)))
    road_graph.add_edges_from(road_graph_edges.keys())

    target_encoded = np.load(f'cache/feature_engineering/{args.city}/eta_labels.npy')

    if args.city == 'london':
        assert target_encoded.shape[0] == 110
    elif args.city == 'madrid':
        assert target_encoded.shape[0] == 109
    elif args.city == 'melbourne':
        assert target_encoded.shape[0] == 108
    else:
        raise ValueError()

    eta_mean = target_encoded.mean(axis=(0, 2))
    t_eta_mean = target_encoded.mean(axis=0)

    if args.city == 'london':
        day_of_week_target_encoded = [
            target_encoded[day_of_week: target_encoded.shape[0]: 7, :, :] for day_of_week in range(7)
        ]
    elif args.city == 'madrid':
        day_of_week_target_encoded = [
            target_encoded[(day_of_week - 1) % 7: target_encoded.shape[0]: 7, :, :] for day_of_week in range(7)
        ]
    elif args.city == 'melbourne':
        day_of_week_target_encoded = [
            target_encoded[day_of_week: target_encoded.shape[0]: 7, :, :] for day_of_week in range(7)
        ]
    else:
        raise ValueError()

    is_weekday_target_encoded = [
        np.concatenate(day_of_week_target_encoded[: 5], axis=0),
        np.concatenate(day_of_week_target_encoded[5:], axis=0)
    ]

    day_of_week_eta_mean = np.stack(
        [day_of_week_target_encoded[day_of_week].mean(axis=(0, 2)) for day_of_week in range(7)],
        axis=0
    )
    day_of_week_t_eta_mean = np.stack(
        [day_of_week_target_encoded[day_of_week].mean(axis=0) for day_of_week in range(7)],
        axis=0
    )

    is_weekday_eta_mean = np.stack(
        [is_weekday_target_encoded[is_weekday].mean(axis=(0, 2)) for is_weekday in range(2)],
        axis=0
    )
    is_weekday_t_eta_mean = np.stack(
        [is_weekday_target_encoded[is_weekday].mean(axis=0) for is_weekday in range(2)],
        axis=0
    )

    test_inputs = read_pickle(path=f'output/test_inputs_extended_{args.city}.pkl')

    features = []

    for test_idx, (supersegment_id, num_nodes, month, day_of_week, t) in enumerate(tqdm(test_inputs)):
        is_weekday = 0 if day_of_week == 5 or day_of_week == 6 else 1
        features.append([
            supersegment_id,
            day_of_week,
            month,
            is_weekday,
            num_nodes,
            eta_mean[supersegment_id],
            is_weekday_eta_mean[is_weekday, supersegment_id],
            day_of_week_eta_mean[day_of_week, supersegment_id],
            t_eta_mean[supersegment_id, t],
            eta_smoothing(
                eta_list=[
                    t_eta_mean[supersegment_id, t - 4],
                    t_eta_mean[supersegment_id, t - 3],
                    t_eta_mean[supersegment_id, t - 2],
                    t_eta_mean[supersegment_id, t - 1],
                    t_eta_mean[supersegment_id, t],
                    t_eta_mean[supersegment_id, t + 1] if t + 1 < 96 else np.nan,
                    t_eta_mean[supersegment_id, t + 2] if t + 2 < 96 else np.nan,
                    t_eta_mean[supersegment_id, t + 3] if t + 3 < 96 else np.nan,
                    t_eta_mean[supersegment_id, t] if t < 96 else np.nan,
                ]
            ),
            is_weekday_t_eta_mean[is_weekday, supersegment_id, t],
            eta_smoothing(
                eta_list=[
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t - 4],
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t - 3],
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t - 2],
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t - 1],
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t],
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t + 1] if t + 1 < 96 else np.nan,
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t + 2] if t + 2 < 96 else np.nan,
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t + 3] if t + 3 < 96 else np.nan,
                    is_weekday_t_eta_mean[is_weekday, supersegment_id, t] if t < 96 else np.nan
                ]
            ),
            day_of_week_t_eta_mean[day_of_week, supersegment_id, t],
            eta_smoothing(
                eta_list=[
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t - 4],
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t - 3],
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t - 2],
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t - 1],
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t],
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t + 1] if t + 1 < 96 else np.nan,
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t + 2] if t + 2 < 96 else np.nan,
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t + 3] if t + 3 < 96 else np.nan,
                    day_of_week_t_eta_mean[day_of_week, supersegment_id, t] if t < 96 else np.nan,
                ]
            ),
            t
        ])

    features = np.array(features, dtype=np.float32)

    return features


def main():
    args = parse_args()

    test_features = preprocess_test(args=args)

    np.save(f'output/test_features_extended_lgb_{args.city}_v2.npy', test_features)


if __name__ == '__main__':
    main()
