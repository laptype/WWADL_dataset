import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录（根据项目结构调整）
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# 将根目录添加到 sys.path
sys.path.insert(0, project_root)
print(f"Added to sys.path: {project_root}")
import csv
import numpy as np
import h5py
import json
from tqdm import tqdm
from manage import WWADLDatasetSplit, WWADLDataset
from utils.h5 import load_h5, save_h5
from utils.h5batch import H5Manager


def save_h5_in_batches(filepath, file_name_list, root_path, time_len, time_step, modality_list=None):
    """
    分批将文件数据保存到 HDF5 文件中。

    Args:
        filepath (str): 输出 HDF5 文件路径。
        file_name_list (list): 要处理的文件名列表。
        root_path (str): 数据集根路径。
        time_len (int): 数据切分的时间长度。
        time_step (int): 数据切分的时间步长。
        modality_list (list): 需要处理的模态列表。
    """
    def normalize_labels(labels):
        """将不规则的标签序列转换为规则数组，使用 dtype=object。"""
        return np.array(labels, dtype=object)

    with h5py.File(filepath, 'w') as h5file:
        for i, file_name in enumerate(tqdm(file_name_list, desc=f"Saving data to {filepath}")):
            dataset = WWADLDataset(root_path, [file_name], modality_list)
            segmented_data = dataset.segment_data(time_len, time_step, target_len=2048)

            for modality, data_dict in segmented_data.items():
                modality_group = h5file.require_group(modality)
                modality_group.create_dataset(f"data_{i}", data=data_dict['data'])
                modality_group.create_dataset(f"label_{i}", data=normalize_labels(data_dict['label']))
    print(f"Data successfully saved to {filepath}")


def process_and_save_dataset_v2(root_path, time_len, time_step, modality_list=None, output_dir='./output', name='',
                                batch_size=100, target_len = 2048, receivers_to_keep = None, new_mapping=None, id2action = None):
    """
    处理数据集并分批保存 data 和 label 到 HDF5 和 JSON 文件中。
    output_dir/name/
    ├── train_data.h5
    ├── train_label.json
    ├── test_data.h5
    ├── test_label.json    output_dir/name/
    ├── train_data.h5
    ├── train_label.json
    ├── test_data.h5
    ├── test_label.json
    ├── train.csv
    ├── test.csv
    └── info.json
    ├── train.csv
    ├── test.csv
    └── info.json
    Args:
        root_path (str): 数据集根目录。
        time_len (int): 时间切片长度。
        time_step (int): 时间切片步长。
        modality_list (list): 数据模态列表。
        output_dir (str): 输出文件夹路径。
        name (str): 文件名。
        batch_size (int): 每批处理的文件数量。
    """
    # 创建输出目录
    output_dir = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集分割
    dataset = WWADLDatasetSplit(root_path=root_path)
    file_records = dataset.generate_file_list()
    train_list, test_list = dataset.split_data_by_volunteer_and_scene(file_records)

    # 导出文件列表为 CSV
    train_csv_path = os.path.join(output_dir, 'train.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')
    dataset.export_to_csv(train_list, train_csv_path)
    dataset.export_to_csv(test_list, test_csv_path)

    # 获取训练和测试的文件名列表
    train_file_name_list, test_file_name_list = dataset.get_file_name_lists(train_list, test_list)

    # 处理并保存训练数据
    print("Processing training data...")
    train_h5_path = os.path.join(output_dir, 'train_data.h5')
    train_json_path = os.path.join(output_dir, 'train_label.json')
    train_dataset = WWADLDataset(root_path, train_file_name_list, modality_list, receivers_to_keep, new_mapping=new_mapping)
    train_data, train_labels = train_dataset.segment_data_h5(time_len, time_step, target_len = target_len, output_file=train_h5_path)
    save_data_in_batches_v2(None, train_labels, batch_size, train_h5_path, train_json_path)

    # 处理并保存测试数据
    print("Processing test data...")
    test_h5_path = os.path.join(output_dir, 'test_data.h5')
    test_json_path = os.path.join(output_dir, 'test_label.json')
    # test_id_json_path = os.path.join(output_dir, 'test_segment.json')
    test_dataset = WWADLDataset(root_path, test_file_name_list, modality_list, receivers_to_keep, new_mapping=new_mapping)
    # test_data, test_labels = test_dataset.segment_data_h5(time_len, time_step, target_len = target_len, output_file=test_h5_path, is_test=True)
    # save_data_in_batches_v2(None, test_labels[0], batch_size, test_h5_path, test_json_path, is_test=True)

    # with open(test_id_json_path, 'w') as json_file:
    #     json.dump(convert_to_serializable(test_labels[1]), json_file, indent=4)
    #     print(test_id_json_path)

    # test_labels = test_labels[0]

    test_dataset.generate_annotations(output_dir, id2action=id2action)

    # 生成 info.json
    generate_info_json_v2(
        output_dir,
        train_data,
        train_labels,
        modality_list,
        time_len,
        time_step,
        {
            "target_len": target_len,
            'train': train_dataset.info,
            'test': train_dataset.info,
            "receivers_to_keep": receivers_to_keep if receivers_to_keep is not None else 'None',
            "id2action": id2action if id2action is not None else 'None',
        }
    )


    print("Processing and saving completed!")

def save_data_in_batches_v2(data, labels, batch_size, test_id_json_path, json_file_path, is_test=False):
    """
    分批次保存数据到 H5 和 JSON 文件中。
    :param data: dict, key 为模态名，value 为对应 np.array，形状为 (n, ...)
    :param labels: dict, key 为模态名，value 为对应的 list 标签
    :param batch_size: 每批保存的样本数量
    :param h5_file_path: 保存 H5 文件的路径
    :param json_file_path: 保存 JSON 文件的路径
    """
    label_dict = {}

    for modality, modality_data in labels.items():
        n = len(modality_data)

        # 分批写入数据
        for i in range(0, n, batch_size):
            start = i
            end = min(i + batch_size, n)

            # 同时保存对应的标签
            for j in range(start, end):
                if modality not in label_dict:
                    label_dict[modality] = {}
                # 将标签值转换为 Python 原生类型
                label_dict[modality][j] = convert_to_serializable(labels[modality][j])

    # 保存标签到 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(label_dict, json_file, indent=4)

def convert_to_serializable(obj):
    """
    将对象转换为 JSON 可序列化的类型。
    :param obj: 任意对象
    :return: JSON 可序列化对象
    """
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def generate_info_json_v2(output_dir, train_data, train_labels, modality_list, time_len, time_step, segment_info = None):
    """
    生成包含数据集信息的 info.json 文件。
    :param output_dir: 输出文件夹路径
    :param train_data: 训练数据
    :param train_labels: 训练标签
    :param test_data: 测试数据
    :param test_labels: 测试标签
    :param modality_list: 数据模态列表
    :param time_len: 时间切片长度
    :param time_step: 时间切片步长
    """
    info = {
        "time_len": time_len,
        "time_step": time_step,
        "modality_list": modality_list,
        "train_data": {
            "num_samples": {modality: data[0] for modality, data in train_data.items()},
            "data_shape": {modality: data[1:] for modality, data in train_data.items()}
        },
        "test_data": {
            "data_shape": {modality: data[1:] for modality, data in train_data.items()}
        },
        "labels": {
            "train": {modality: len(train_labels[modality]) for modality in train_labels},
        }
    }

    if segment_info is not None:
        info['segment_info'] = segment_info

    info_json_path = os.path.join(output_dir, 'info.json')
    with open(info_json_path, 'w') as json_file:
        json.dump(info, json_file, indent=4)

def read_data_by_id(h5_file_path, json_file_path, sample_id):
    """
    根据指定的 ID 读取数据和标签。
    :param h5_file_path: H5 文件的路径
    :param json_file_path: JSON 文件的路径
    :param sample_id: 要读取的样本 ID
    :return: 对应 ID 的数据和标签
    """
    # 读取标签
    with open(json_file_path, 'r') as json_file:
        labels = json.load(json_file)

    if str(sample_id) not in labels:
        raise ValueError(f"ID {sample_id} 不存在！")

    # 读取 HDF5 数据
    with h5py.File(h5_file_path, 'r') as h5_file:
        data = h5_file['data'][sample_id]  # 按索引读取数据

    label = labels[str(sample_id)]
    return data, label


device2name = {
    'gl': 'glasses',
    'lh': 'left hand',
    'rh': 'right hand',
    'lp': 'left pocket',
    'rp': 'right pocket'
}

# Example usage
if __name__ == "__main__":

    from dataset_management.action import old_to_new_mapping, new_id_to_action

    root_path = '/root/shared-nvme/WWADL'
    time_len = 30
    time_step = 3
    modality_list = ['imu', 'wifi']  # 需要处理的模态
    output_dir = '/root/shared-nvme/dataset'


    name = f'all_{time_len}_{time_step}'
    process_and_save_dataset_v2(
        root_path,
        time_len,
        time_step,
        modality_list,
        output_dir,
        name,
        target_len=2048,
        receivers_to_keep=None,
        new_mapping=old_to_new_mapping,
        id2action=new_id_to_action
    )

    root_path = '/root/shared-nvme/WWADL'
    time_len = 30
    time_step = 3
    modality_list = ['imu']  # 需要处理的模态
    output_dir = '/root/shared-nvme/dataset'
    # name = 'wifi'

    # for device in ['gl', 'lh', 'rh', 'lp', 'rp']:
    #
    #     name = f'imu_{time_len}_{time_step}_{device}'
    #     receivers_to_keep = {
    #         'imu': [device2name[device]]
    #     }
    #     process_and_save_dataset_v2(
    #         root_path,
    #         time_len,
    #         time_step,
    #         modality_list,
    #         output_dir,
    #         name,
    #         target_len=2048,
    #         receivers_to_keep=receivers_to_keep
    #     )


    #
    # '''
    #     target_len 是最后缩放了的结果,因为网络输入需要2048
    # '''

    # root_path = '/root/shared-nvme/WWADL'
    # # root_path = '/data/WWADL/processed_data'
    # time_len = 30
    # time_step = 3
    # modality_list = ['wifi']  # 需要处理的模态
    # output_dir = '/root/shared-nvme/dataset'
    # name = 'wifi'
    #
    #
    # for i in range(3):
    #     name = f'wifi_{time_len}_{time_step}_{i}'
    #     receivers_to_keep = {
    #         'wifi': [i]
    #     }
    #
    #     process_and_save_dataset_v2(
    #         root_path,
    #         time_len,
    #         time_step,
    #         modality_list,
    #         output_dir,
    #         name,
    #         target_len=2048,
    #         receivers_to_keep = receivers_to_keep
    #     )

