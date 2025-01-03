
from dataset_management.base import WWADLBase
from utils.h5 import load_h5

class WWADL_wifi(WWADLBase):
    def __init__(self, file_path):
        super().__init__()
        self.load_data(file_path)

    def load_data(self, file_path):

        data = load_h5(file_path)

        self.data = data['amp']
        self.label = data['label']


    def segment(self, time_len = 30, sample_rate = 200):

        self.data   # (11400, 3, 3, 30)
        window_len = time_len
        self.label # [[id, action_type, start_id, end_id], ....]










