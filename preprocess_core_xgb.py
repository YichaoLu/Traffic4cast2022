import argparse

import networkx as nx
import numpy as np
from tqdm import tqdm

from file_utils import read_pickle

highway_dict = {
    'primary': 0,
    'secondary': 1,
    'tertiary': 2,
    'trunk': 3,
    'motorway': 4,
    'residential': 5,
    'primary_link': 6,
    'secondary_link': 7,
    'tertiary_link': 8,
    'trunk_link': 9,
    'motorway_link': 10,
    'unclassified': 11
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def additive_smoothing(
        empirical_count: np.ndarray,
        empirical_mean: np.ndarray,
        pseudo_count: int = 20
):
    return (empirical_count + empirical_mean * pseudo_count) / (np.sum(empirical_count) + pseudo_count)


def preprocess_test(args: argparse.Namespace):
    node_id_dict = read_pickle(path=f'cache/road_graph/{args.city}/node_id_dict.pkl')
    road_graph_edges = read_pickle(path=f'cache/road_graph/{args.city}/road_graph_edges.pkl')

    edge_id_dict = read_pickle(path=f'cache/feature_engineering/{args.city}/edge_id_dict.pkl')
    cc_labels_counter = np.load(f'cache/feature_engineering/{args.city}/cc_labels_counter.npy')
    t_cc_labels_counter = np.load(f'cache/feature_engineering/{args.city}/t_cc_labels_counter.npy')
    weekday_cc_labels_counter = np.load(f'cache/feature_engineering/{args.city}/weekday_cc_labels_counter.npy')
    day_of_week_cc_labels_counter = np.load(f'cache/feature_engineering/{args.city}/day_of_week_cc_labels_counter.npy')
    weekday_t_cc_labels_counter = np.load(f'cache/feature_engineering/{args.city}/weekday_t_cc_labels_counter.npy')
    day_of_week_t_cc_labels_counter = np.load(f'cache/feature_engineering/{args.city}/day_of_week_t_cc_labels_counter.npy')

    road_graph = nx.DiGraph()
    road_graph.add_nodes_from(range(len(node_id_dict)))
    road_graph.add_edges_from(road_graph_edges.keys())

    if args.city == 'london':
        empirical_mean = np.array([0.5367906303432076, 0.35138063340805714, 0.11182873624873524], dtype=np.float32)
    elif args.city == 'madrid':
        empirical_mean = np.array([0.4976221039083026, 0.3829591430424158, 0.1194187530492816], dtype=np.float32)
    elif args.city == 'melbourne':
        empirical_mean = np.array([0.7018930324884697, 0.2223245729555099, 0.0757823945560204], dtype=np.float32)
    else:
        raise ValueError()

    for u, v in road_graph_edges:
        edge_attributes = list(road_graph_edges[(u, v)])
        highway, lanes, tunnel = edge_attributes[3], edge_attributes[5], edge_attributes[6]
        highway_one_hot = [0] * 12
        if ',' in highway:
            highway = eval(highway)
            for highway_attr in highway:
                highway_one_hot[highway_dict[highway_attr]] = 1
        else:
            highway_one_hot[highway_dict[highway]] = 1
        edge_attributes[3] = highway_one_hot
        if len(lanes) == 0:
            edge_attributes[5] = np.nan
        elif ',' in lanes:
            edge_attributes[5] = float(eval(lanes)[0])
        else:
            edge_attributes[5] = float(lanes)
        if len(tunnel) == 0:
            edge_attributes[6] = [0, 0, 0]
        else:
            if tunnel == 'yes':
                edge_attributes[6] = [1, 0, 0]
            elif tunnel == 'covered':
                edge_attributes[6] = [0, 1, 0]
            elif tunnel == 'building_passage':
                edge_attributes[6] = [0, 0, 1]
            elif tunnel == 'no':
                edge_attributes[6] = [0, 0, 0]
            else:
                tunnel = eval(tunnel)
                assert isinstance(tunnel, list) and len(tunnel) == 2 and tunnel[0] == 'yes' and tunnel[1] == 'building_passage'
                edge_attributes[6] = [1, 0, 1]
        if args.city == 'madrid' or args.city == 'melbourne':
            edge_attributes[4] = eval(edge_attributes[4])
            if isinstance(edge_attributes[4], list):
                edge_attributes[4] = False
        road_graph_edges[(u, v)] = tuple(edge_attributes)

    test_inputs = read_pickle(path=f'output/test_inputs_{args.city}.pkl')

    features = []

    for test_idx, (u, v, u_volumes_1h, v_volumes_1h, month, day_of_week, t) in enumerate(tqdm(test_inputs)):
        is_weekday = 0 if day_of_week == 5 or day_of_week == 6 else 1

        edge_id = edge_id_dict[(u, v)]
        (
            parsed_maxspeed,
            speed_kph,
            importance,
            highway,
            oneway,
            lanes,
            tunnel,
            length_meters,
            counter_distance
        ) = road_graph_edges[(u, v)]
        u_sources = [source for source, sink in road_graph.in_edges(u)]
        v_sources = [source for source, sink in road_graph.in_edges(v)]
        u_sinks = [sink for source, sink in road_graph.out_edges(u)]
        v_sinks = [sink for source, sink in road_graph.out_edges(v)]
        static_features = [
            u,
            v,
            edge_id,
            day_of_week,
            month,
            is_weekday,
            parsed_maxspeed,
            speed_kph,
            importance,
            length_meters,
            counter_distance,
            len(u_sources),
            len(v_sources),
            len(u_sinks),
            len(v_sinks),
            *additive_smoothing(empirical_count=cc_labels_counter[edge_id, :], empirical_mean=empirical_mean),
            *additive_smoothing(empirical_count=weekday_cc_labels_counter[is_weekday, edge_id, :], empirical_mean=empirical_mean),
            *additive_smoothing(empirical_count=day_of_week_cc_labels_counter[day_of_week, edge_id, :], empirical_mean=empirical_mean),
            *highway,
            oneway,
            lanes,
            *tunnel
        ]
        dynamic_features = [
            *u_volumes_1h,
            *v_volumes_1h,
            *additive_smoothing(empirical_count=t_cc_labels_counter[t, edge_id, :], empirical_mean=empirical_mean),
            *additive_smoothing(empirical_count=weekday_t_cc_labels_counter[is_weekday, t, edge_id, :], empirical_mean=empirical_mean),
            *additive_smoothing(empirical_count=day_of_week_t_cc_labels_counter[day_of_week, t, edge_id, :], empirical_mean=empirical_mean),
            t
        ]
        features.append(static_features + dynamic_features)

    features = np.array(features, dtype=np.float32)

    return features


def main():
    args = parse_args()

    test_features = preprocess_test(args=args)

    np.save(f'output/test_features_core_xgb_{args.city}.npy', test_features)


if __name__ == '__main__':
    main()
