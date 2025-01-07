import os.path
import numpy as np
import csv
from tqdm import tqdm
from collections import defaultdict
from utils.h5 import load_h5, save_h5
from dataset_management.wifi import WWADL_wifi
from dataset_management.imu import WWADL_imu
from dataset_management.airpods import WWADL_airpods

class WWADLDataset():

    def __init__(self, root_path, file_name_list, modality_list=None):

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
            self.data['wifi'] = [WWADL_wifi(os.path.join(self.data_path['wifi'], f)) for f in
                                 tqdm(file_name_list, desc="WiFi files")]

        if 'imu' in modality_list:
            print("Loading IMU data...")
            self.data['imu'] = [WWADL_imu(os.path.join(self.data_path['imu'], f)) for f in
                                tqdm(file_name_list, desc="IMU files")]

        if 'airpods' in modality_list:
            print("Loading AirPods data...")
            self.data['airpods'] = [WWADL_airpods(os.path.join(self.data_path['airpods'], f)) for f in
                                    tqdm(file_name_list, desc="AirPods files")]

        self.modality_list = modality_list

        self.sample_rate = { 'wifi': 50, 'imu': 50, 'airpods': 25 }

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
    # file_name_list = [
    #     '2_1_1.h5',
    #     '2_1_10.h5',
    #     '2_1_11.h5'
    # ]
    # dataset = WWADLDataset('/data/WWADL/processed_data', file_name_list, ['imu'])
    # # dataset.check_data()
    #
    # data, label = dataset.segment_data()
    #
    # for k, v in data.items():
    #     print(k, v.shape)
    #
    # # print(data['wifi']['label'])
    # from utils.h5 import save_h5
    #
    # save_h5('test.h5', data)


    data = load_h5('test.h5')

    print(data)