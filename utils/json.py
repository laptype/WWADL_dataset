import json

def save_labels_as_json(filepath, labels):
    """
    将标签保存为 JSON 文件。

    Args:
        filepath (str): 保存路径。
        labels (list): 标签数据。
    """
    with open(filepath, 'w') as f:
        json.dump(labels, f)
    print(f"Labels saved to {filepath}")

def load_labels_from_json(filepath):
    """
    从 JSON 文件加载标签。

    Args:
        filepath (str): 标签文件路径。

    Returns:
        list: 标签数据。
    """
    with open(filepath, 'r') as f:
        labels = json.load(f)
    return labels
