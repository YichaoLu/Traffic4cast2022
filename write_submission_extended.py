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
        supersegment_id_dict = read_pickle(path=f'cache/road_graph/{city}/supersegment_id_dict.pkl')
        supersegment_id_reverse_dict = dict()
        for supersegment_id in supersegment_id_dict:
            supersegment_id_reverse_dict[supersegment_id_dict[supersegment_id]] = supersegment_id

        test_predictions = np.load(f'output/test_predictions_extended_lgb_{city}.npy')

        num_supersegment_ids = len(supersegment_id_dict)

        assert len(test_predictions) == num_supersegment_ids * 100

        identifier_list, test_idx_list, eta_list = [], [], []

        index = 0
        for test_idx in trange(100):
            for supersegment_id in range(num_supersegment_ids):
                identifier_list.append(supersegment_id_reverse_dict[supersegment_id])
                test_idx_list.append(test_idx)
                eta_list.append(test_predictions[test_idx * num_supersegment_ids + supersegment_id])
                index += 1

        df = pd.DataFrame({
            'identifier': identifier_list,
            'test_idx': test_idx_list,
            'eta': eta_list
        })

        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'cache/test/{city}/labels/eta_labels_test.parquet', compression='snappy')

    with zipfile.ZipFile('output/eta_submission.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for city in cities:
            z.write(
                filename=f'cache/test/{city}/labels/eta_labels_test.parquet',
                arcname=os.path.join(city, 'labels', 'eta_labels_test.parquet')
            )


if __name__ == '__main__':
    main()
