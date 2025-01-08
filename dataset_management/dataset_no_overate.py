import os
import pandas as pd
import numpy as np
from manage import WWADLDatasetSplit, WWADLDataset
from dataset_split import save_data_in_batches


def main(root_path, dataset_path, time_len, time_step, target_len, modality_list = None):

    # 初始化 WWADLDataset
    if modality_list is None:
        modality_list = ['imu']

    # 读取 test.csv
    test_csv_path = os.path.join(dataset_path, 'test.csv')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"{test_csv_path} does not exist.")

    print("Loading test.csv...")
    test_df = pd.read_csv(test_csv_path)
    file_name_list = test_df['file_name'].tolist()
    print(f"Loaded {len(file_name_list)} file names from test.csv.")

    print("Processing test data...")
    test_dataset = WWADLDataset(root_path, file_name_list, modality_list)
    test_data, test_labels = test_dataset.segment_data(time_len, time_step, target_len = target_len)



    print(f"Annotations have been saved to {dataset_path}.")

if __name__ == '__main__':
    root_path = '/data/WWADL/processed_data'
    main(root_path,
         '/data/WWADL/dataset/imu_32_3',
         time_len=32,
         time_step=3,
         modality_list=['imu'],
         )
