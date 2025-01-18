"""
{
    "database": {
        "video1": {
            "subset": "validation",
            "annotations": [
                {"segment": [10, 20], "label": "action1"},
                {"segment": [30, 40], "label": "action2"}
            ]
        },
        "video2": {
            "subset": "training",
            "annotations": [
                {"segment": [5, 15], "label": "action3"}
            ]
        }
    }
}
"""
import os
import pandas as pd
import numpy as np
from manage import WWADLDatasetSplit, WWADLDataset


def main(root_path, dataset_path, modality_list = None):
    # 读取 test.csv
    test_csv_path = os.path.join(dataset_path, 'train.csv')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"{test_csv_path} does not exist.")

    print("Loading test.csv...")
    test_df = pd.read_csv(test_csv_path)
    file_name_list = test_df['file_name'].tolist()
    print(f"Loaded {len(file_name_list)} file names from test.csv.")

    # 初始化 WWADLDataset
    if modality_list is None:
        modality_list = ['imu']

    dataset = WWADLDataset(root_path, file_name_list, modality_list)
    # 执行 generate_annotations
    dataset.generate_annotations(dataset_path, label='train')

    print(f"Annotations have been saved to {dataset_path}.")


if __name__ == '__main__':
    root_path = '/root/shared-nvme/WWADL'
    main(root_path, '/root/shared-nvme/dataset/all_30_3', modality_list=['imu'])




