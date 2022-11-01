import argparse

import networkx as nx
import numpy as np
from tqdm import tqdm

from file_utils import read_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    return parser.parse_args()


def preprocess_test(args: argparse.Namespace):
    node_id_dict = read_pickle(path=f'cache/road_graph/{args.city}/node_id_dict.pkl')
    road_graph_edges = read_pickle(path=f'cache/road_graph/{args.city}/road_graph_edges.pkl')

    road_graph = nx.DiGraph()
    road_graph.add_nodes_from(range(len(node_id_dict)))
    road_graph.add_edges_from(road_graph_edges.keys())

    test_inputs = read_pickle(path=f'output/test_inputs_extended_{args.city}.pkl')

    features = []

    for test_idx, (supersegment_id, num_nodes, month, day_of_week, t) in enumerate(tqdm(test_inputs)):
        features.append([
            supersegment_id,
            day_of_week,
            month,
            num_nodes,
            t
        ])

    features = np.array(features, dtype=np.float32)

    return features


def main():
    args = parse_args()

    test_features = preprocess_test(args=args)

    np.save(f'output/test_features_extended_lgb_{args.city}.npy', test_features)


if __name__ == '__main__':
    main()
