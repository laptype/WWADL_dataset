import os
import sys

import pandas as pd
from dataset_management.manage import WWADLDatasetSplit, WWADLDataset
from dataset_management.action import old_to_new_mapping, new_id_to_action

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录（根据项目结构调整）
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# 将根目录添加到 sys.path
sys.path.insert(0, project_root)
print(f"Added to sys.path: {project_root}")


def load_file_list(dataset_path, mode='train'):
    # 读取 test.csv
    test_csv_path = os.path.join(dataset_path, f"{mode}.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"{test_csv_path} does not exist.")

    print("Loading test.csv...")
    test_df = pd.read_csv(test_csv_path)
    file_name_list = test_df['file_name'].tolist()
    print(f"Loaded {len(file_name_list)} file names from {mode}.csv.")
    return file_name_list


def get_airpods_spilt_dataset(root_path, dataset_path):

    new_mapping = old_to_new_mapping
    id2action = new_id_to_action

    target_len = 2048

    time_len = 30
    time_step = 3

    train_datalist = load_file_list(dataset_path, mode='train')
    test_datalist = load_file_list(dataset_path, mode='test')

    print(len(train_datalist))
    print(len(test_datalist))

    modality_list = ['imu', 'wifi', 'airpods']

    print("Processing training data...")
    train_h5_path = os.path.join(dataset_path, 'train_data_airpods.h5')
    train_json_path = os.path.join(dataset_path, 'train_label_airpods.json')
    train_dataset = WWADLDataset(root_path, train_datalist, modality_list, receivers_to_keep=None, new_mapping=new_mapping)
    train_data, train_labels = train_dataset.segment_data_h5(time_len, time_step, target_len = target_len, output_file=train_h5_path)

    

if __name__ == '__main__':
    dataset_path = '/root/shared-nvme/dataset/all_30_3'
    root_path = '/data/WWADL/processed_data'
    get_airpods_spilt_dataset(root_path, dataset_path)



