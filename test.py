import h5py

data_path = '/root/shared-nvme/dataset/imu_30_3/test_data.h5'
modality = 'imu'
with h5py.File(data_path, 'r') as h5_file:
    data = h5_file['imu']['0_1_15.h5']
    print(data.keys())