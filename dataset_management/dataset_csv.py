import os.path


class WWADL_Dataset():
    def __init__(self, root_path = '/data/WWADL/processed_data'):
        self.imu_path = os.path.join(root_path, 'imu')
        self.wifi_path = os.path.join(root_path, 'wifi')
        self.airpods_path = os.path.join(root_path, 'AirPodsPro')





