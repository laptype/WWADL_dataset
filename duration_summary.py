import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录（根据项目结构调整）
project_root = os.path.abspath(os.path.join(current_dir))
# 将根目录添加到 sys.path
sys.path.insert(0, project_root)
print(f"Added to sys.path: {project_root}")

import csv
import numpy as np
import h5py
import pandas as pd
import json
from tqdm import tqdm
from dataset_management.manage import WWADLDatasetSplit, WWADLDataset
from utils.h5 import load_h5, save_h5
from utils.h5batch import H5Manager
from dataset_management.action import old_to_new_mapping, new_id_to_action

def convert_seconds_to_hours(seconds):
    hours = int(seconds // 3600)
    remaining_seconds = seconds % 3600
    minutes = int(remaining_seconds // 60)
    seconds = int(remaining_seconds % 60)
    return hours, minutes, seconds

def sum_durations(duration1, duration2):
    total_seconds = 0

    # 累加两个字典中每个场景的时长（单位：秒）
    for scene_id in duration1:
        total_seconds += duration1[scene_id]
        total_seconds += duration2.get(scene_id, 0)

    # 将总秒数转换为小时、分钟和秒
    total_hours, total_minutes, total_seconds = convert_seconds_to_hours(total_seconds)
    return total_hours, total_minutes, total_seconds


def load_file_list(dataset_path, mode='train'):
    # 读取 test.csv
    test_csv_path = os.path.join(dataset_path, f"{mode}.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"{test_csv_path} does not exist.")

    print("Loading test.csv...")
    test_df = pd.read_csv(test_csv_path)
    file_name_list = test_df['file_name'].tolist()
    print(f"Loaded {len(file_name_list)} file names from {mode}.csv.")
    return file_name_list


def duration_summary(test_train_csv_path, root_path, modality_list=['imu']):
    # 获取训练和测试的文件名列表
    train_file_name_list = load_file_list(test_train_csv_path, mode='train')
    test_file_name_list = load_file_list(test_train_csv_path, mode='test')

    train_dataset = WWADLDataset(root_path, train_file_name_list, modality_list, new_mapping=old_to_new_mapping)
    train_duration = train_dataset.duration_summary()
    train_count = train_dataset.count_summary()
    test_dataset = WWADLDataset(root_path, test_file_name_list, modality_list, new_mapping=old_to_new_mapping)
    test_duration = test_dataset.duration_summary()
    test_count = test_dataset.count_summary()

    print(train_duration)

    # 输出训练集每个场景的总时长
    print("Train dataset duration per scene:")
    train_scene_total = {}  # 用来存储每个场景在训练集中的总时长
    for scene_id, duration in train_duration.items():
        hours, minutes, seconds = convert_seconds_to_hours(duration)
        train_scene_total[scene_id] = (hours, minutes, seconds)
        print(f"Scene {scene_id}: {hours} hours, {minutes} minutes, {seconds} seconds")

    # 输出测试集每个场景的总时长
    print("\nTest dataset duration per scene:")
    test_scene_total = {}  # 用来存储每个场景在测试集中的总时长
    for scene_id, duration in test_duration.items():
        hours, minutes, seconds = convert_seconds_to_hours(duration)
        test_scene_total[scene_id] = (hours, minutes, seconds)
        print(f"Scene {scene_id}: {hours} hours, {minutes} minutes, {seconds} seconds")

    # 计算并输出训练集每个场景的总时长（训练集和测试集加起来）
    print("\nTotal duration per scene (Train + Test):")
    for scene_id in train_scene_total:
        train_hours, train_minutes, train_seconds = train_scene_total[scene_id]
        test_hours, test_minutes, test_seconds = test_scene_total.get(scene_id, (0, 0, 0))
        
        # 累加训练集和测试集的时长
        total_seconds = (train_hours * 3600 + train_minutes * 60 + train_seconds) + (test_hours * 3600 + test_minutes * 60 + test_seconds)
        
        # 转换为小时、分钟和秒
        total_hours, total_minutes, total_seconds = convert_seconds_to_hours(total_seconds)
        
        print(f"Scene {scene_id}: {total_hours} hours, {total_minutes} minutes, {total_seconds} seconds")

    # 计算并输出训练集的总时长
    train_total_seconds = sum(train_duration.values())  # 所有场景的秒数总和
    train_hours, train_minutes, train_seconds = convert_seconds_to_hours(train_total_seconds)
    print(f"\nTotal duration of train dataset: {train_hours} hours, {train_minutes} minutes, {train_seconds} seconds")

    # 计算并输出测试集的总时长
    test_total_seconds = sum(test_duration.values())  # 所有场景的秒数总和
    test_hours, test_minutes, test_seconds = convert_seconds_to_hours(test_total_seconds)
    print(f"Total duration of test dataset: {test_hours} hours, {test_minutes} minutes, {test_seconds} seconds")

    # 计算并输出训练集和测试集的总时长
    total_hours, total_minutes, total_seconds = sum_durations(train_duration, test_duration)
    print(f"\nTotal duration of train and test datasets: {total_hours} hours, {total_minutes} minutes, {total_seconds} seconds")

    # Output the count of instances for each scene in the training dataset
    print("Train dataset count per scene:")
    for scene_id, count in train_count.items():
        print(f"Scene {scene_id}: {count} instances")

    # Output the count of instances for each scene in the testing dataset
    print("\nTest dataset count per scene:")
    for scene_id, count in test_count.items():
        print(f"Scene {scene_id}: {count} instances")

    # Output the total count for each scene (Train + Test)
    print("\nTotal count per scene (Train + Test):")
    for scene_id in train_count:
        train_scene_count = train_count[scene_id]
        test_scene_count = test_count.get(scene_id, 0)
        
        # Calculate total count for each scene
        total_count = train_scene_count + test_scene_count
        print(f"Scene {scene_id}: {total_count} instances")

    # Other calculations and print statements as previously
    # Output train dataset total count
    train_total_count = sum(train_count.values())
    print(f"\nTotal count of train dataset: {train_total_count} instances")

    # Output test dataset total count
    test_total_count = sum(test_count.values())
    print(f"Total count of test dataset: {test_total_count} instances")

    # Output train and test dataset total count
    total_count = train_total_count + test_total_count
    print(f"\nTotal count of train and test datasets: {total_count} instances")

if __name__ == '__main__':

    test_train_csv_path = '/root/shared-nvme/dataset/all_30_3'
    root_path = '/root/shared-nvme/WWADL'
    output_dir = '/root/shared-nvme/dataset'
    duration_summary(test_train_csv_path, root_path=root_path, modality_list=['imu'])

