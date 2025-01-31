import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from dataset_management.action import id_to_action

def handle_nan_and_interpolate(data, window_len, target_len):
    """
    插值并在插值前处理 NaN 值的通用函数。
    Args:
        data (np.ndarray): 输入数据，维度为 (window_len, ...)
        window_len (int): 原始时序长度。
        target_len (int): 目标时序长度。
    Returns:
        np.ndarray: 插值后的数据，时间维度变为 target_len，其他维度保持不变。
    """
    original_shape = data.shape  # 获取原始形状
    flattened_data = data.reshape(window_len, -1)  # 展平除时间维度以外的所有维度

    # 定义插值函数
    def interpolate_channel(channel_data):
        original_indices = np.linspace(0, window_len - 1, window_len)
        target_indices = np.linspace(0, window_len - 1, target_len)

        # 检查 NaN 并处理
        nan_mask = np.isnan(channel_data)
        if np.any(nan_mask):  # 如果存在 NaN 值
            valid_indices = np.where(~nan_mask)[0]
            valid_values = channel_data[~nan_mask]

            if len(valid_indices) > 1:  # 至少两个有效值
                interp_func = interp1d(valid_indices, valid_values, kind='linear', bounds_error=False, fill_value="extrapolate")
                channel_data = interp_func(np.arange(window_len))
            else:
                # 如果有效值不足，填充为 0
                channel_data = np.zeros_like(channel_data)

        # 插值到目标长度
        interp_func = interp1d(original_indices, channel_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        return interp_func(target_indices)

    # 对所有通道进行插值处理
    interpolated_flattened_data = np.array([
        interpolate_channel(flattened_data[:, i])
        for i in range(flattened_data.shape[1])
    ]).T  # 转置回时间维度在前

    # 恢复原始形状，时间维度替换为 target_len
    reshaped_interpolated_data = interpolated_flattened_data.reshape(target_len, *original_shape[1:])
    return reshaped_interpolated_data

class WWADLBase():
    def __init__(self, file_path):
        self.data = None
        self.label = None
        self.file_name = os.path.basename(file_path)
        self.window_len = 0
        self.window_step = 0

    def load_data(self, file_path):
        pass

    def show_info(self):
        print(self.data.shape)
        print(self.label)

    def __len__(self):
        return self.data.shape[0]

    def segment(self, time_len=30, step=3, sample_rate=200, target_len=2048, is_test=False):
        """
        滑动窗口切分数据并生成对应的标注 label
        Args:
            time_len: 每个窗口的时间长度（秒）
            step: 滑动窗口的步长（秒）
            sample_rate: 采样率（每秒采样点数）
        Returns:
            segmented_data: 切分后的数据 (num_windows, ...)
            targets: 每个窗口对应的目标标签
        """

        # 计算窗口长度和滑动步长
        window_len = time_len * sample_rate
        window_step = step * sample_rate

        self.window_len = window_len
        self.window_step = window_step

        if self.data.shape[0] < window_len:
            raise ValueError(f"Data length ({self.data.shape[0]}) is less than window length ({window_len}).")

        # 滑动窗口切分数据
        segmented_data = []
        targets = []
        # print(self.data.shape)
        test_target = []

        # 创建 offsetlist 确保最后一个片段不被遗漏
        offsetlist = list(range(0, self.data.shape[0] - window_len + 1, window_step))
        if (self.data.shape[0] - window_len) % window_step:
            offsetlist.append(self.data.shape[0] - window_len)

        for start in offsetlist:
            end = start + window_len
            test_target.append([start, end])
            window_data = self.data[start:end]

            # 调用封装的函数进行处理
            interpolated_data = handle_nan_and_interpolate(window_data, window_len, target_len)
            assert not np.any(np.isnan(interpolated_data)), "Data contains NaN values!"

            # # 替换 NaN 为 0
            # window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
            #
            # # 插值到 target_len 长度
            # original_indices = np.linspace(0, window_len - 1, window_len)
            # target_indices = np.linspace(0, window_len - 1, target_len)
            # interpolated_data = interp1d(original_indices, window_data, axis=0, kind='linear')(target_indices)

            segmented_data.append(interpolated_data)

            # 处理 label 生成目标标签
            window_start_time = start
            window_end_time = end
            window_targets = []

            for lbl in self.label:
                obj_start_time = lbl[2]
                obj_end_time = lbl[3]
                obj_label = lbl[1]

                # 计算目标在窗口内的交集
                intersection_start = max(window_start_time, obj_start_time)
                intersection_end = min(window_end_time, obj_end_time)
                intersection_duration = max(0, intersection_end - intersection_start)
                obj_duration = obj_end_time - obj_start_time

                # 判断交集是否覆盖目标时间的 80%
                if intersection_duration / obj_duration >= 0.8:
                    # 计算相对时间位置
                    relative_start = max(0, obj_start_time - window_start_time) / window_len
                    relative_end = min(window_len, obj_end_time - window_start_time) / window_len
                    window_targets.append([relative_start, relative_end, obj_label])

            # 如果窗口内有目标，则添加到 targets
            if window_targets:
                targets.append(window_targets)
        # 转换为 np.array
        segmented_data = np.array(segmented_data)

        if is_test:
            return segmented_data, (targets, test_target)
        else:
            return segmented_data, targets

    def mapping_label(self, old_to_new_mapping):
        for i in range(len(self.label)):
            self.label[i][1] = old_to_new_mapping[self.label[i][1]]

    def generate_annotations(self, subset, id2action: dict = None):
        """
        根据标签数据生成JSON格式的标注文件。

        参数:
            subset (str): 数据集的子集名称，例如 'validation' 或 'training'。

        返回:
            dict: JSON格式的标注数据。
        """
        if id2action is None:
            id2action = id_to_action

        if self.label is None:
            raise ValueError("Label data is not set.")

        data_len = len(self.data)

        annotations = []
        for row in self.label:
            action_id = row[1]
            start_id = row[2]
            end_id = row[3]
            annotations.append({
                "segment": [start_id, end_id],
                "label": f"{id2action[action_id]}"
            })

        json_data = {
            "subset": subset,
            "annotations": annotations,
            "data_shape": str(self.data.shape)
        }
        return (self.file_name, json_data)
