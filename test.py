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


# with h5py.File('/root/shared-nvme/WWADL/wifi/0_1_15.h5', 'r') as h5_file:
#     for key in h5_file.keys():
#         data = np.array(h5_file[key])
#         print(data)
        # if np.isnan(data).any():
        #     print(f"NaN detected in dataset {key}")

import pandas as pd

# Define the data for the table
data = {
    "User": ["user1", "user2", "user2", "user4", "user5", "user6", "user7", "user8", "user9", "user10", "user11", "user12", "user13", "user14", "user15", "user16"],
    "mAP@0.5": [94.31, 90.08, 96.91, 96.97, 55.42, 92.63, 91.15, 90.76, 93.44, 89.21, 95.71, 92.55, 95.45, 91.23, 86.24, 93.38],
    "0.55": [94.29, 88.89, 96.91, 96.61, 55.30, 92.16, 89.98, 90.64, 93.12, 88.21, 95.38, 92.09, 95.45, 91.22, 85.77, 92.69],
    "0.6": [93.96, 87.84, 96.49, 95.57, 54.79, 90.86, 89.12, 90.07, 91.71, 87.14, 94.69, 90.98, 94.81, 90.65, 85.07, 91.00],
    "0.65": [93.91, 84.72, 96.14, 94.62, 53.31, 89.57, 87.78, 89.41, 90.70, 84.68, 93.30, 89.84, 94.20, 88.31, 82.19, 88.75],
    "0.7": [93.04, 80.64, 92.21, 90.71, 51.18, 87.91, 85.84, 84.24, 87.47, 77.40, 89.27, 85.21, 91.45, 84.83, 77.43, 85.44],
    "0.75": [90.85, 75.15, 88.25, 85.03, 48.45, 86.37, 82.33, 78.33, 83.60, 72.43, 85.40, 78.49, 87.77, 81.50, 68.54, 81.84],
    "0.8": [84.06, 61.94, 84.93, 76.84, 44.80, 83.59, 75.20, 71.25, 76.70, 61.59, 79.47, 71.52, 81.00, 74.48, 64.21, 73.44],
    "0.85": [74.31, 44.69, 75.87, 61.34, 36.64, 74.51, 62.78, 58.67, 60.40, 45.25, 65.21, 56.59, 68.42, 62.79, 49.70, 60.77],
    "0.9": [53.53, 27.68, 57.34, 41.90, 23.37, 61.00, 47.43, 39.68, 41.77, 31.87, 44.65, 35.82, 47.25, 46.23, 36.31, 47.54],
    "0.95": [43.54, 20.13, 45.85, 25.27, 14.67, 45.90, 39.66, 31.31, 32.18, 24.79, 35.75, 25.18, 34.38, 36.23, 27.63, 35.82],
    "mAP@avg": [81.58, 66.18, 83.09, 76.49, 43.79, 80.45, 75.13, 72.44, 75.11, 66.26, 77.88, 71.83, 79.02, 74.75, 66.31, 75.07]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate the mean for each column (ignoring the first 'User' column)
mean_values = df.iloc[:, 1:].mean()

# Add the average row to the table
df.loc["Average"] = mean_values

print(df.loc["Average"])