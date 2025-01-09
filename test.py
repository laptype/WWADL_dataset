import h5py
import numpy as np
# data_path = '/root/shared-nvme/dataset/imu_30_3/test_data.h5'
# # modality = 'imu'
# # with h5py.File(data_path, 'r') as h5_file:
# #     data = h5_file['imu']['0_1_15.h5']
# #     print(data.keys())
#
# def check_h5_file(data_path, modality):
#     with h5py.File(data_path, 'r') as h5_file:
#         data = h5_file[modality]
#         print(f"Checking dataset {modality} in {data_path}...")
#
#         for idx in range(len(data)):
#             sample = np.array(data[idx])
#             if np.isnan(sample).any():
#                 print(f"NaN detected in sample {idx}")
#             if np.isinf(sample).any():
#                 print(f"Inf detected in sample {idx}")
#
#         print("Finished checking dataset.")
#
# if __name__ == '__main__':
#     # data_path = '/root/shared-nvme/dataset/wifi_30_3/train_data.h5'
#     # check_h5_file(data_path, 'wifi')
#
#     data_path = '/root/shared-nvme/dataset/imu_30_3/train_data.h5'
#     check_h5_file(data_path, 'imu')


with h5py.File('/root/shared-nvme/WWADL/wifi/0_1_15.h5', 'r') as h5_file:
    for key in h5_file.keys():
        data = np.array(h5_file[key])
        print(data)
        # if np.isnan(data).any():
        #     print(f"NaN detected in dataset {key}")