import os
import json
import h5py
import numpy as np


def load_h5(filepath):
    def recursively_load_group_to_dict(h5file, path):
        """
        递归加载 HDF5 文件中的组和数据集为嵌套字典
        """
        result = {}
        group = h5file[path]

        for key, item in group.items():
            if isinstance(item, h5py.Group):
                # 如果是组，则递归加载
                result[key] = recursively_load_group_to_dict(h5file, f"{path}/{key}")
            elif isinstance(item, h5py.Dataset):
                # 如果是数据集，则加载为 NumPy 数组
                result[key] = item[()]

        return result

    with h5py.File(filepath, 'r') as h5file:
        return recursively_load_group_to_dict(h5file, '/')


def save_h5(filepath, data):
    def recursively_save_dict_to_group(h5file, path, dictionary):
        """
        递归保存嵌套字典到 HDF5 文件
        """
        for key, value in dictionary.items():
            # 处理键的路径
            full_path = f"{path}/{key}"

            if isinstance(value, dict):
                # 如果是字典，则递归保存
                recursively_save_dict_to_group(h5file, full_path, value)
            elif isinstance(value, np.ndarray):
                # 如果是 NumPy 数组，直接保存
                h5file.create_dataset(full_path, data=value)
            else:
                # 如果是其他类型，转换为 NumPy 数组后保存
                h5file.create_dataset(full_path, data=np.array(value))

    with h5py.File(filepath, 'w') as h5file:
        recursively_save_dict_to_group(h5file, '/', data)