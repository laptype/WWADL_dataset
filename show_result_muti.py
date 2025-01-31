"""
IoU 和 NMS：
    iou 计算两个片段的重叠度。
    nms 删除低分数的冗余检测。
"""
import os
import json
import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端，适用于没有显示器的环境
import matplotlib.pyplot as plt
import numpy as np


# IoU 计算函数
def iou(segment1, segment2):
    start1, end1 = segment1
    start2, end2 = segment2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union


# NMS 函数
def nms(results, iou_thresh=0.5):
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    final_results = []
    while sorted_results:
        best = sorted_results.pop(0)  # 获取置信度最高的框
        final_results.append(best)
        sorted_results = [
            result for result in sorted_results
            if iou(best['segment'], result['segment']) < iou_thresh
        ]  # 删除与最高置信度框 IoU 过高的框
    return final_results

all_labels = set()
color_map = {}
# {'Reading': (0.19215686274509805, 0.5098039215686274, 0.7411764705882353, 1.0), 'Using Phone': (0.6196078431372549, 0.792156862745098, 0.8823529411764706, 1.0), 'Sitting Down': (0.9019607843137255, 0.3333333333333333, 0.050980392156862744, 1.0), 'Walking': (0.9921568627450981, 0.6823529411764706, 0.4196078431372549, 1.0), 'Lying Down': (0.19215686274509805, 0.6392156862745098, 0.32941176470588235, 1.0), 'Pouring Water': (0.7803921568627451, 0.9137254901960784, 0.7529411764705882, 1.0), 'Taking Medicine': (0.6196078431372549, 0.6039215686274509, 0.7843137254901961, 1.0), 'Getting Out of Bed': (0.8549019607843137, 0.8549019607843137, 0.9215686274509803, 1.0), 'Stretching': (0.5882352941176471, 0.5882352941176471, 0.5882352941176471, 1.0), 'Turning On/Off Eye Protection Lamp': (0.6196078431372549, 0.6039215686274509, 0.7843137254901961, 1.0), 'Writing': (0.8549019607843137, 0.8549019607843137, 0.9215686274509803, 1.0), 'Standing Up': (0.38823529411764707, 0.38823529411764707, 0.38823529411764707, 1.0), 'Opening Envelope': (0.7411764705882353, 0.7411764705882353, 0.7411764705882353, 1.0)}


# Predefined set of labels
labels = [
    'Pouring Water', 'Stretching', 'Getting Out of Bed', 'Taking Medicine', 
    'Lying Down', 'Using Phone', 'Reading', 'Walking', 'Sitting Down', 
    'Turning On/Off Eye Protection Lamp', 'Opening Envelope', 'Writing', 'Standing Up', 'Washing Hands', 'Picking Fruit', 'Cutting Fruit', 'Eating Fruit', 'Throwing Garbage'
]

label2label = {
    'Pouring Water': 'Pour water',
    'Stretching': 'Stretching',
    'Getting Out of Bed': 'Get up',
    'Taking Medicine': 'Take medicine',
    'Lying Down': 'Lie down',
    'Using Phone': 'Use phone',
    'Reading': 'Read a book',
    'Walking': 'Walk',
    'Sitting Down': 'Sit down',
    'Turning On/Off Eye Protection Lamp': 'on/off the desk lamp',
    'Opening Envelope': 'Open an envelope',
    'Writing': 'Write',
    'Standing Up': 'Stand up',
    'Washing Hands': 'Wash hands',
    'Picking Fruit': 'Pick up things',
    'Cutting Fruit': 'Cut fruits',
    'Eating Fruit': 'Eat fruits',
    'Throwing Garbage': 'Throw waste'
}


# Generate a set of distinct colors using a colormap with enough distinct colors
num_labels = len(labels)

# You can use colormap 'tab20' which has 20 distinct colors
colors = plt.cm.tab20b(np.linspace(0, 1, num_labels))  # This will give you 'num_labels' distinct colors
# colors = [(r * 0.9, g * 0.9, b * 0.9, a) for r, g, b, a in colors]  # Multiply RGB by 0.7 to darken

# Map each label to a unique color
color_map = dict(zip(labels, colors))


# Print the resulting color map to check
for label, color in color_map.items():
    print(f"{label}: {color}")

# def update_color_map():
#     """
#     Update the global color map based on the unique labels.
#     Each label is assigned a unique color from the color map.
#     """
#     global color_map
#     # Assign a new color for each label if it's not already assigned
#     for label in all_labels:
#         if label not in color_map:  # Only add label if not already in color_map
#             # Generate a unique color for each label
#             color_map[label] = plt.cm.tab20c(len(color_map) / len(all_labels))  # Use 'tab20c' colormap for more distinct colors


def visualize_multiple_results(video_name, groundtruth, predictions_list, method_names, title="Action Detection Results", save_path=None, is_label=True):
    """
    Visualize Groundtruth and Predictions for multiple methods with Y-axis representing methods.
    
    Parameters:
    - video_name: str, video name to visualize
    - groundtruth: list of dict, each dict contains 'segment' and 'label'
    - predictions_list: list of lists, each sublist contains predictions for one method
    - method_names: list of str, names of the methods corresponding to predictions_list
    - title: str, title of the plot
    - save_path: str, path to save the plot (optional)
    """
    global all_labels, color_map  # Refer to the global variables

    # Update the global 'all_labels' with new labels from the current video
    all_labels.update([gt['label'] for gt in groundtruth])
    for preds in predictions_list:
        all_labels.update([pred['label'] for pred in preds])

    # Update the color map whenever all_labels changes
    # update_color_map()

    # Setup the plot
    plt.figure(figsize=(16, 5))

    # Plot Groundtruth
    y_groundtruth = len(predictions_list)  # Groundtruth at the top
    for gt in groundtruth:
        start, end = gt['segment']
        label = gt['label']
        label_dis = label2label[label]
        # Display the label and IoU score above the segment
        if label == 'Turning On/Off Eye Protection Lamp':
            label_dis = 'On/Off Lamp'
        plt.plot([start, end], [y_groundtruth, y_groundtruth], color=color_map[label], linewidth=8, label=f"GT: {label_dis}")
        plt.text((start + end) / 2, y_groundtruth + 0.2, f"{label_dis}", ha='center', va='bottom', fontsize=10, color=color_map[label], fontweight='bold')

    # Plot each method's predictions
    for i, (predictions, method_name) in enumerate(zip(predictions_list, method_names)):
        y_level = i  # Each method's predictions at a separate y-level
        for pred in predictions:
            start, end = pred['segment']
            label = pred['label']
            
            # Find corresponding ground truth (assuming one-to-one matching based on label)
            corresponding_gt = next((gt for gt in groundtruth if gt['label'] == label), None)
            if corresponding_gt:
                gt_segment = corresponding_gt['segment']
                # Calculate IoU between the prediction and the ground truth
                iou_score = iou(pred['segment'], gt_segment)
            else:
                iou_score = 0  # No ground truth found for this label, IoU is 0
            if iou_score < 0.05:
                continue
            # Plot the prediction segment
            plt.plot([start, end], [y_level, y_level], color=color_map[label], linewidth=8, label=f"{method_name}: {label} (IoU: {iou_score:.2f})")
            label_dis = label2label[label]
            # Display the label and IoU score above the segment
            if label == 'Turning On/Off Eye Protection Lamp':
                label_dis = 'On/Off Lamp'
            plt.text((start + end) / 2, y_level + 0.1 + 0.3, f"{label_dis}", 
                     ha='center', va='bottom', fontsize=10, color=color_map[label], fontweight='bold')
            plt.text((start + end) / 2, y_level + 0.1, f"{iou_score:.2f}", 
                     ha='center', va='bottom', fontsize=9, color=color_map[label])

    # Format plot
    # plt.title(f"(b) an visualization example of office", fontsize=14)
    # plt.title(, fontsize=14)
    plt.title(title, fontsize=14)
    if is_label:
        plt.xlabel("Time (s)", fontsize=14)
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], ['0', '10', '20', '30', '40', '50', '60'], fontsize=12)
    y_ticks = list(range(len(predictions_list) + 1))
    y_labels = method_names + ["Groundtruth"]
    plt.yticks(y_ticks, y_labels, fontsize=14, fontweight='bold')
    plt.ylim(-1, len(predictions_list) + 1)  # Add space for better visualization
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=False)

    # Show or save the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{video_name}.svg"))
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# 后处理函数
def post_process(raw_results, conf_thresh=0.3, iou_thresh=0.3, top_k=100, use_nms=True):
    """
    Post-process raw predictions using confidence thresholding and NMS.

    conf_thresh
    含义：
        conf_thresh 是 置信度阈值，用于筛选检测结果。
        模型为每个检测结果生成一个置信度分数（score），表示该结果属于某类别的可能性。conf_thresh 定义了一个下限，低于该阈值的检测结果会被剔除。
        如果 conf_thresh 设置得太低，可能会保留许多低置信度的预测，导致冗余或错误结果。
        如果 conf_thresh 设置得太高，可能会丢失一些正确的预测，特别是低置信度但正确的检测。
    设置建议：
        通常设置为 0.5 或 0.6。如果希望只保留非常高置信度的结果，可以设置为 0.7 或更高。

    iou_thresh
    含义：
        iou_thresh 是 交并比（IoU）阈值，用于在 非极大值抑制（NMS） 中判断两个检测结果是否冗余。
        IoU（Intersection over Union）衡量两个检测区间的重叠程度：
        iou_thresh 定义了一个上限，IoU 大于该阈值的结果被认为是冗余的。
    作用：
        删除高度重叠的检测结果，避免重复预测。
        保留分数最高的检测结果，删除分数较低的重叠结果。

    top_k
        含义：

        top_k 是一个限制值，用于控制每类检测结果的数量上限。
        即使经过 NMS 或后处理后，仍可能有较多的检测结果，top_k 限制最终保留的结果数量。
        作用：

        避免存储过多的检测结果，减少计算和存储的开销。
        保留分数最高的 top_k 个结果，确保输出结果的质量

    Returns:
    - list of dict, processed results
    """
    # Step 1: Filter by confidence threshold
    filtered_results = [res for res in raw_results if res['score'] >= conf_thresh]

    # Step 2: Apply NMS
    if use_nms:
        filtered_results = nms(filtered_results, iou_thresh=iou_thresh)

    # Step 3: Keep Top-K results
    filtered_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)[:top_k]

    return filtered_results


# 示例数据
groundtruth_path = '/root/shared-nvme/dataset/all_30_3/imu_annotations.json'
# json_file_path = "/root/shared-nvme/code_result/result/25_01-10/test2/WWADLDatasetSingle_imu_30_3_34_2048_30_0/checkpoint_wifiTAD_34_2048_30_0-epoch-20.pt.json"  # Replace with the actual file path

# 文件路径
mamba = "/root/shared-nvme/code_result/result/25_01-23/fusion_backbone/WWADLDatasetMuti_all_30_3_mamba_layer_8_i_1/checkpoint_mamba_mamba_layer_8_i_1-epoch-77.pt.json"
wifiTAD = "/root/shared-nvme/code_result/result/25_01-21/muti_w/WWADLDatasetMuti_all_30_3_wifiTAD/checkpoint_wifiTAD_wifiTAD-epoch-79.pt.json"
actionformer = "/root/shared-nvme/code_result/result/25_01-22/ActionFormer/WWADLDatasetMuti_all_30_3_ActionFormer_layer_8_i_1/checkpoint_Transformer_ActionFormer_layer_8_i_1-epoch-77.pt.json"
actionmamba = "/root/shared-nvme/code_result/result/25_01-22/ActionMamba/WWADLDatasetMuti_all_30_3_ActionMamba_layer_8_i_1/checkpoint_mamba_ActionMamba_layer_8_i_1-epoch-78.pt.json"
tempmax = "/root/shared-nvme/code_result/result/25_01-22/TemporalMaxer/WWADLDatasetMuti_all_30_3_TemporalMaxer_i_1/checkpoint_TemporalMaxer_TemporalMaxer_i_1-epoch-79.pt.json"
tridet = "/root/shared-nvme/code_result/result/25_01-22/TriDet/WWADLDatasetMuti_all_30_3_TriDet_i_2/checkpoint_TriDet_TriDet_i_2-epoch-79.pt.json"
unet = "/root/shared-nvme/code_result/result/25_01-19/single_imu/WWADLDatasetSingle_all_30_3_Transformer_layer_8_embed_type_Norm/checkpoint_Transformer_Transformer_layer_8_embed_type_Norm-epoch-79.pt.json"
groundtruth_path = "/root/shared-nvme/dataset/all_30_3/imu_annotations.json"

# 读取所有的预测文件
# paths = [mamba, wifiTAD, actionformer, actionmamba, tempmax, tridet, unet]
paths = [tempmax, unet, tridet, actionmamba, actionformer, wifiTAD, mamba]
with open(mamba, "r") as f:
    predictions = json.load(f)

# for video_name in ['0_1_3.h5', '0_2_10.h5', '2_3_2.h5', '2_3_6.h5']:
for video_name in ['0_1_3.h5', '9_2_13.h5', '3_3_2.h5']:
# for video_name in predictions['results'].keys():
    predictions_list = []
    if video_name == '9_2_13.h5':
        paths = [tempmax, unet, tridet, actionmamba, actionformer, wifiTAD, mamba]
        title = f"(b) a visualization example of Study room"
        is_label = False
    if video_name == '0_1_3.h5':
        paths = [tempmax, actionformer, tridet, actionmamba, unet, wifiTAD, mamba]
        title = f"(a) a visualization example of Bedroom"
        is_label = False
    if video_name == '3_3_2.h5':
        paths = [tempmax, tridet, unet, actionmamba, actionformer, wifiTAD, mamba]
        title = f"(c) a visualization example of Dining room"   
        is_label = True

    for path in paths:
        with open(path, "r") as f:
            predictions = json.load(f)
        raw_predictions = predictions['results'][video_name]  # 假设使用文件中的`'results'`字段，注意要根据实际情况调整
        predictions_list.append(post_process(raw_predictions, conf_thresh=0.7, iou_thresh=0.02, top_k=10))

    # 方法名称
    # method_names = [
    #     "XRFMamba",
    #     "WiFiTAD",
    #     "ActionFormer",
    #     "ActionMamba",
    #     "TemporalMaxer",
    #     "TriDet",
    #     "Unet"
    # ]
    method_names = [
        "TemporalMaxer",
        "UWiFiAction",
        "TriDet",
        "ActionFormer",
        "ActionMamba",
        "WiFiTAD",
        "XRFMamba",
    ]

    # 读取Groundtruth数据
    with open(groundtruth_path, "r") as f:
        groundtruth_json = json.load(f)

    groundtruth = groundtruth_json['database'][video_name]['annotations']

    # 可视化多个方法的结果
    save_path = os.path.dirname(mamba)  # 使用`mamba`路径获取保存位置
    save_path = os.path.join('/root/shared-nvme/code_result/result', 'img')
    visualize_multiple_results(video_name, groundtruth, predictions_list, method_names, save_path=save_path, title=title, is_label= is_label)

print(all_labels)