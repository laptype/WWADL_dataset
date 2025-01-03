import os.path

from utils.h5 import load_h5
from dataset_management.wifi import WWADL_wifi
from dataset_management.imu import WWADL_imu
from dataset_management.airpods import WWADL_airpods


class WWADLDataset():

    def __init__(self, root_path, file_name_list):

        self.data_path = {
            'wifi': os.path.join(root_path, 'wifi'),
            'imu': os.path.join(root_path, 'imu'),
            'airpods': os.path.join(root_path, 'AirPodsPro'),
        }

        self.data = {
            'wifi': [WWADL_wifi(os.path.join(self.data_path['wifi'], f)) for f in file_name_list],
            'imu': [WWADL_imu(os.path.join(self.data_path['imu'], f)) for f in file_name_list],
            'airpods': [WWADL_airpods(os.path.join(self.data_path['airpods'], f)) for f in file_name_list],
        }

    def check_data(self):
        print(self.data['wifi'][0].data.shape)
        print(self.data['imu'][0].duration)
        print(self.data['wifi'][0].label)


if __name__ == '__main__':
    file_name_list = [
        '2_1_1.h5',
        '2_1_10.h5',
        '2_1_11.h5'
    ]
    dataset = WWADLDataset('/data/WWADL/processed_data', file_name_list)
    dataset.check_data()