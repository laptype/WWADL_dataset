import os.path
import numpy as np
import csv
import json
import h5py
from tqdm import tqdm
from collections import defaultdict
from utils.h5 import load_h5, save_h5
from dataset_management.wifi import WWADL_wifi
from dataset_management.imu import WWADL_imu
from dataset_management.airpods import WWADL_airpods

class WWADLDataset():

    def __init__(self, root_path, file_name_list, modality_list=None, receivers_to_keep=None, new_mapping=None):

        if receivers_to_keep is None:
            receivers_to_keep = {
                "imu": None,
                "wifi": None,
                "airpods": None
            }
        if modality_list is None:
            modality_list = ['wifi', 'imu', 'airpods']
        self.data_path = {
            'wifi': os.path.join(root_path, 'wifi'),
            'imu': os.path.join(root_path, 'imu'),
            'airpods': os.path.join(root_path, 'AirPodsPro'),
        }

        # Initialize an empty dictionary for data
        self.data = {}

        # Only include modalities specified in the `modality` list
        if 'wifi' in modality_list:
            print("Loading WiFi data...")
            self.data['wifi'] = [WWADL_wifi(os.path.join(self.data_path['wifi'], f),
                                            receivers_to_keep['wifi'],
                                            new_mapping=new_mapping) for f in
                                 tqdm(file_name_list, desc="WiFi files")]

        if 'imu' in modality_list:
            print("Loading IMU data...")
            self.data['imu'] = [WWADL_imu(os.path.join(self.data_path['imu'], f),
                                          receivers_to_keep['imu'],
                                          new_mapping=new_mapping) for f in
                                tqdm(file_name_list, desc="IMU files")]

        if 'airpods' in modality_list:
            print("Loading AirPods data...")
            self.data['airpods'] = [WWADL_airpods(os.path.join(self.data_path['airpods'], f)) for f in
                                    tqdm(file_name_list, desc="AirPods files")]

        self.modality_list = modality_list

        self.sample_rate = { 'wifi': 50, 'imu': 50, 'airpods': 25 }

        self.info = {}

    def segment_data(self, time_len = 30, time_step = 3, target_len=2048):
        """
        对所有文件的数据进行切分，并整理为 np.array
        Returns:
            segmented_data: 包含所有切分后的数据，格式为字典：
                            {
                                'wifi': {'data': np.array, 'label': list},
                                'imu': {'data': np.array, 'label': list},
                                'airpods': {'data': np.array, 'label': list}
                            }
        """
        segmented_data = {}
        segmented_label = {}
        for modality in self.modality_list:
            all_data = []
            all_labels = []

            # 获取采样率
            sample_rate = self.sample_rate[modality]

            # 遍历每个文件的实例，添加进度条
            print(f"Processing modality: {modality}")
            for instance in tqdm(self.data[modality], desc=f"Processing {modality} files"):
                # 切分数据
                data, labels = instance.segment(time_len, time_step, sample_rate, target_len=target_len)

                # print(data.shape)
                # 将结果追加到列表中
                all_data.append(data)
                all_labels.extend(labels)

            # 将每个模态的数据和标签合并为单个 np.array
            segmented_data[modality] = np.concatenate(all_data, axis=0)  # 合并数据
            segmented_label[modality] = all_labels

        return segmented_data, segmented_label

    def segment_data_h5(self, time_len=30, time_step=3, target_len=2048, output_file='segmented_data.h5', chunk_size=1,
                        is_test=False):
        """
        对所有文件的数据进行切分，将 data 存储到 HDF5 文件中，并将 label 保留在内存中。
        Args:
            time_len: 每段数据的时间长度
            time_step: 时间步长
            target_len: 目标长度
            output_file: 保存 HDF5 文件的路径
            chunk_size: HDF5 数据块大小
            is_test: 是否为测试模式，测试模式下每个样本单独保存
        Returns:
            segmented_label: 包含所有标签的字典
        """
        segmented_label = {}
        data_shapes = {}  # 用于存储每个模态数据的最终维度
        with h5py.File(output_file, 'w') as h5f:
            for modality in self.modality_list:
                print(f"Processing modality: {modality}")

                all_labels = []
                test_all_labels = {}
                # 遍历每个文件的实例并处理数据
                for instance in tqdm(self.data[modality], desc=f"Processing {modality} files"):
                    data, labels = instance.segment(
                        time_len, time_step, self.sample_rate[modality], target_len=target_len, is_test=is_test
                    )
                    self.info['window_len'] = instance.window_len
                    self.info['window_step'] = instance.window_step

                    if is_test:
                        # 测试模式：每条样本单独保存
                        for sample_index, sample_data in enumerate(data):
                            dataset_name = f'{modality}/{instance.file_name}/{sample_index}'
                            h5f.create_dataset(
                                dataset_name,
                                data=sample_data,
                                dtype=sample_data.dtype,
                                chunks=sample_data.shape
                            )
                        all_labels.extend(labels[0])  # 保存标签
                        test_all_labels[f"{instance.file_name}"] = labels[1]
                    else:
                        # 非测试模式：合并保存数据
                        if modality not in h5f:
                            # 第一次遇到该模态，创建数据集
                            modality_data_shape = (0,) + data.shape[1:]
                            max_shape = (None,) + data.shape[1:]
                            data_dset = h5f.create_dataset(
                                f'{modality}',
                                shape=modality_data_shape,
                                maxshape=max_shape,
                                dtype=data.dtype,
                                chunks=(chunk_size,) + data.shape[1:]
                            )
                        else:
                            # 获取已存在的数据集
                            data_dset = h5f[f'{modality}']

                        # 扩展数据集
                        current_data_size = data_dset.shape[0]
                        new_data_size = current_data_size + data.shape[0]
                        data_dset.resize(new_data_size, axis=0)
                        data_dset[current_data_size:new_data_size] = data

                        # 保存标签
                        all_labels.extend(labels)

                # 保存每个模态的标签和数据形状
                segmented_label[modality] = all_labels
                if not is_test and modality in h5f:
                    data_shapes[modality] = h5f[f'{modality}'].shape  # 保存最终数据维度
                print(f"Finished processing modality: {modality}")

        print(f"Data saved to {output_file}")
        if is_test:
            return data_shapes, (segmented_label, test_all_labels)
        else:
            return data_shapes, segmented_label


    def generate_annotations(self, save_path, id2action = None):
        for modality in self.modality_list:
            print(f"Processing modality: {modality}")
            all_json_data = {
                "database": {}
            }
            for instance in tqdm(self.data[modality], desc=f"Processing {modality} files"):
                key, json_data = instance.generate_annotations("test", id2action=id2action)
                all_json_data["database"][key] = convert_to_serializable(json_data)

            # 保存 JSON 文件
            modality_save_path = f"{save_path}/{modality}_annotations.json"
            with open(modality_save_path, 'w') as json_file:
                json.dump(all_json_data, json_file, indent=4)
            print(f"Saved JSON file for modality '{modality}' to {modality_save_path}")


    def check_data(self):
        # print(self.data['wifi'][0].data.shape)
        # print(self.data['imu'][0].duration)
        print(self.data['wifi'][0].label)

        data, label = self.data['wifi'][0].segment(30, 3, 200)

        print(data.shape)
        print(label)

        print(self.data['imu'][0].data.shape)
        data, label = self.data['imu'][0].segment(30, 3, 50)

        print(data.shape)
        print(label)

        print(self.data['airpods'][0].data.shape)

        data, label = self.data['airpods'][0].segment(30, 3, 25)

        print(data.shape)
        print(label)

def convert_to_serializable(obj):
    """
    将对象转换为 JSON 可序列化的格式。
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

class WWADLDatasetSplit:
    def __init__(self, root_path='/data/WWADL/processed_data'):
        self.root_path = root_path
        self.imu_path = os.path.join(root_path, 'imu')
        self.wifi_path = os.path.join(root_path, 'wifi')
        self.airpods_path = os.path.join(root_path, 'AirPodsPro')

    def generate_file_list(self, volunteer_ids=range(16), scene_ids=range(1, 4), action_ids=range(1, 21)):
        file_records = []

        # Iterate through volunteer IDs, scene IDs, and action IDs
        for volunteer_id in volunteer_ids:  # Default: 0-15
            for scene_id in scene_ids:  # Default: 1-3
                for action_id in action_ids:  # Default: 1-20
                    # Generate the file name
                    file_name = f"{volunteer_id}_{scene_id}_{action_id}.h5"

                    # Check file existence in each folder
                    imu_file = os.path.join(self.imu_path, file_name)
                    wifi_file = os.path.join(self.wifi_path, file_name)
                    airpods_file = os.path.join(self.airpods_path, file_name)

                    if os.path.exists(imu_file) and os.path.exists(wifi_file) and os.path.exists(airpods_file):
                        # Append record if all files are found
                        file_records.append({
                            'volunteer_id': volunteer_id,
                            'scene_id': scene_id,
                            'action_group_id': action_id,
                            'imu_path': imu_file,
                            'wifi_path': wifi_file,
                            'airpods_path': airpods_file,
                            'file_name': file_name
                        })

        return file_records

    def split_data_by_volunteer_and_scene(self, file_records, train_ratio=0.8):
        # Organize files by volunteer_id and scene_id
        data_dict = defaultdict(lambda: defaultdict(list))
        for record in file_records:
            volunteer_id = record['volunteer_id']
            scene_id = record['scene_id']
            data_dict[volunteer_id][scene_id].append(record)

        # Split into train and test lists
        train_list = []
        test_list = []

        for volunteer_id, scenes in data_dict.items():
            for scene_id, records in scenes.items():
                # Shuffle the data for randomness
                np.random.shuffle(records)

                # Calculate split index
                split_index = int(len(records) * train_ratio)

                # Split into training and testing
                train_list.extend(records[:split_index])
                test_list.extend(records[split_index:])

        return train_list, test_list

    def export_to_csv(self, data_list, file_name):
        # Define CSV headers
        headers = ['volunteer_id', 'scene_id', 'action_group_id', 'imu_path', 'wifi_path', 'airpods_path', 'file_name']

        save_path = os.path.join(self.root_path, file_name)

        # Write to CSV
        with open(save_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_list)

        print(f"Data exported to {save_path}")

    def get_file_name_lists(self, train_list, test_list):
        # Extract file_name from train_list and test_list
        train_file_name_list = [record['file_name'] for record in train_list]
        test_file_name_list = [record['file_name'] for record in test_list]

        return train_file_name_list, test_file_name_list



if __name__ == '__main__':
    file_name_list = [
        '2_1_1.h5',
        '2_1_10.h5',
        '2_1_11.h5'
    ]
    from dataset_management.action import old_to_new_mapping, new_id_to_action
    dataset = WWADLDataset('/root/shared-nvme/WWADL/', file_name_list, ['imu'], new_mapping=old_to_new_mapping)
    # dataset.check_data()
    data = dataset.data['imu']
    # data, label = dataset.segment_data()


    print(new_id_to_action)

    # for k in data:
    #     print(k.data.shape)

    # for k, v in data.items():
    #     print(k, v.shape)
    #

    # # print(data['wifi']['label'])
    # from utils.h5 import save_h5
    #
    # save_h5('test.h5', data)

    #
    # data = load_h5('test.h5')
    #
    # print(data)