import h5py
import numpy as np
import os
import ast


class H5Manager:
    def __init__(self, file_path):
        """
        初始化 H5Manager 实例。

        Args:
            file_path (str): HDF5 文件路径。
        """
        self.file_path = file_path
        # if not os.path.exists(file_path):
        with h5py.File(file_path, 'w') as f:
            # 改为不使用压缩选项
            f.create_dataset("index_map", data=np.string_("{}"))
            print(f"Created new HDF5 file at {file_path}")

    def save(self, data_dict, batch_size=100):
        """
        保存数据到 HDF5 文件，支持追加批次。

        Args:
            data_dict (dict): 包含数据的字典，格式为 {modality: np.array}.
            batch_size (int): 每批数据存储的大小。
        """
        with h5py.File(self.file_path, 'a') as h5file:
            # 检查或初始化索引表
            if "index_map" not in h5file:
                h5file.create_dataset("index_map", data=np.string_("{}"))
                index_map = {}
            else:
                try:
                    index_map = ast.literal_eval(h5file["index_map"][()].decode("utf-8"))
                except (SyntaxError, ValueError):
                    print("Error parsing index_map, reinitializing...")
                    index_map = {}

            for modality, data in data_dict.items():
                print(f"Saving modality: {modality}")
                group = h5file.require_group(modality)
                indices = index_map.get(modality, [])
                start_idx = indices[-1][1] if indices else 0

                # 按批次保存数据
                for i in range(0, len(data), batch_size):
                    end_idx = min(i + batch_size, len(data))
                    dataset_name = f"data_{len(indices)}"
                    group.create_dataset(dataset_name, data=data[i:end_idx], compression="gzip")
                    indices.append((start_idx, start_idx + (end_idx - i)))
                    start_idx += (end_idx - i)

                # 更新索引表
                index_map[modality] = indices

            h5file["index_map"][...] = np.string_(str(index_map))
        print(f"Data saved to {self.file_path}")

    def show_shapes(self):
        """
        显示 HDF5 文件中每个模态的所有数据集的形状。
        """
        with h5py.File(self.file_path, 'r') as h5file:
            print(f"File structure for {self.file_path}:")
            for modality in h5file.keys():
                if modality == "index_map":
                    continue
                print(f"  {modality}:")
                group = h5file[modality]
                for dataset_name in group.keys():
                    print(f"    {dataset_name}: shape={group[dataset_name].shape}")

    def query_by_id(self, modality, target_id):
        """
        查询指定模态中某个 ID 的数据。

        Args:
            modality (str): 要查询的模态（如 'wifi', 'imu', 'airpods'）。
            target_id (int): 要查询的全局 ID。

        Returns:
            np.array: 对应的目标数据。
        """
        with h5py.File(self.file_path, 'r') as h5file:
            index_map = ast.literal_eval(h5file["index_map"][()].decode("utf-8"))
            if modality not in index_map:
                raise ValueError(f"Modality '{modality}' not found in {self.file_path}.")

            indices = index_map[modality]
            group = h5file[modality]
            for i, (start, end) in enumerate(indices):
                if start <= target_id < end:
                    dataset_name = f"data_{i}"
                    local_id = target_id - start
                    return group[dataset_name][local_id]

        raise ValueError(f"ID {target_id} not found in modality '{modality}'.")

    def get_index_map(self):
        """
        获取当前文件的全局索引表。

        Returns:
            dict: 索引表，格式为 {modality: [(start, end), ...]}.
        """
        with h5py.File(self.file_path, 'r') as h5file:
            return ast.literal_eval(h5file["index_map"][()].decode("utf-8"))
