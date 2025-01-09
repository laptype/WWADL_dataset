
from dataset_management.base import WWADLBase
from utils.h5 import load_h5
import numpy as np

class WWADL_wifi(WWADLBase):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.load_data(file_path)
        # sample = self.data
        # if np.isnan(sample).any():
        #     print(f"NaN detected in sample")
        # if np.isinf(sample).any():
        #     print(f"Inf detected in sample")

    def load_data(self, file_path):

        data = load_h5(file_path)

        self.data = data['amp']

        sample = self.data

        self.label = data['label']














