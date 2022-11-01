import os
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import trange

from file_utils import read_pickle


def main():
    cities = ['london', 'madrid', 'melbourne']

    for city in cities:
        node_id_reverse_dict = read_pickle(path=f'cache/road_graph/{city}/node_id_reverse_dict.pkl')
        road_graph_edges = read_pickle(path=f'cache/road_graph/{city}/road_graph_edges.pkl')
        sorted_road_graph_edges = sorted(road_graph_edges.keys())

        test_predictions = np.load(f'output/test_predictions_core_xgb_{city}.npy')

        test_predictions = np.log(test_predictions)

        assert test_predictions.shape == (len(sorted_road_graph_edges) * 100, 3)

        u_list, v_list, test_idx_list, logit_green_list, logit_yellow_list, logit_red_list = [], [], [], [], [], []

        index = 0
        for test_idx in trange(100):
            for u, v in sorted_road_graph_edges:
                u_list.append(node_id_reverse_dict[u])
                v_list.append(node_id_reverse_dict[v])
                test_idx_list.append(test_idx)
                logit_green_list.append(test_predictions[index, 0])
                logit_yellow_list.append(test_predictions[index, 1])
                logit_red_list.append(test_predictions[index, 2])
                index += 1

        df = pd.DataFrame({
            'u': u_list,
            'v': v_list,
            'test_idx': test_idx_list,
            'logit_green': logit_green_list,
            'logit_yellow': logit_yellow_list,
            'logit_red': logit_red_list
        })

        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'cache/test/{city}/labels/cc_labels_test.parquet', compression='snappy')

    with zipfile.ZipFile('output/cc_submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for city in cities:
            z.write(
                filename=f'cache/test/{city}/labels/cc_labels_test.parquet',
                arcname=os.path.join(city, 'labels', 'cc_labels_test.parquet')
            )


if __name__ == '__main__':
    main()
