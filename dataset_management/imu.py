from dataset_management.base import WWADLBase
from utils.h5 import load_h5


class WWADL_imu(WWADLBase):
    def __init__(self, file_path):
        super().__init__()
        self.duration = 0
        self.load_data(file_path)


    def load_data(self, file_path):
        data = load_h5(file_path)

        self.data = data['data']
        self.label = data['label']
        self.duration = data['duration']



