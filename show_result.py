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


# 可视化函数
def visualize_results(video_name, groundtruth, predictions, title="Action Detection Results"):
    """
    Visualize Groundtruth and Predictions for a video.

    Parameters:
    - video_name: str, video name to visualize
    - groundtruth: list of dict, each dict contains 'segment' and 'label'
    - predictions: list of dict, each dict contains 'segment', 'label', and 'score'
    """
    # Get unique labels for consistent coloring
    all_labels = set([gt['label'] for gt in groundtruth] + [pred['label'] for pred in predictions])
    color_map = {label: plt.cm.tab20(i / len(all_labels)) for i, label in enumerate(all_labels)}

    # Setup the plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

    # Plot Groundtruth
    for gt in groundtruth:
        start, end = gt['segment']
        label = gt['label']
        axes[0].plot([start, end], [0, 0], color=color_map[label], linewidth=8, label=label)
    axes[0].set_title("Groundtruth")
    axes[0].set_yticks([])
    axes[0].legend(loc="upper right")

    # Plot Predictions
    for pred in predictions:
        start, end = pred['segment']
        label = pred['label']
        score = pred['score']
        axes[1].plot([start, end], [0, 0], color=color_map[label], linewidth=8, label=f"{label} ({score:.2f})")
    axes[1].set_title("Predictions")
    axes[1].set_yticks([])
    axes[1].legend(loc="upper right")

    # Finalize plot
    plt.suptitle(f"{title} for {video_name}")
    plt.xlabel("Time")
    plt.tight_layout()
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


def visualize_results_combined(video_name, groundtruth, predictions, title="Action Detection Results", save_path=None):
    """
    Visualize Groundtruth and Predictions for a video on the same plot.

    Parameters:
    - video_name: str, video name to visualize
    - groundtruth: list of dict, each dict contains 'segment' and 'label'
    - predictions: list of dict, each dict contains 'segment', 'label', and 'score'
    """
    # Get unique labels for consistent coloring
    all_labels = set([gt['label'] for gt in groundtruth] + [pred['label'] for pred in predictions])
    color_map = {label: plt.cm.tab20(i / len(all_labels)) for i, label in enumerate(all_labels)}

    # Setup the plot
    plt.figure(figsize=(15, 6))

    # Plot Groundtruth
    for gt in groundtruth:
        start, end = gt['segment']
        label = gt['label']
        plt.plot([start, end], [1, 1], color=color_map[label], linewidth=8, label=f"GT: {label}")

    # Plot Predictions
    for pred in predictions:
        start, end = pred['segment']
        label = pred['label']
        score = pred['score']
        plt.plot([start, end], [0, 0], color=color_map[label], linewidth=8, label=f"Pred: {label} ({score:.2f})")

    # Format plot
    plt.title(f"{title} for {video_name}", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.yticks([0, 1], ["Predictions", "Groundtruth"], fontsize=12)
    plt.ylim(-1, 2)  # 设置 y 轴范围
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=False)

    # Show plot
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'result.jpg'))  # Save the plot as a file
        print(f"Plot saved to {save_path}")
    else:
        plt.show()  # Show the plot if no save_path is provided



# 示例数据
groundtruth_path = '/root/shared-nvme/dataset/all_30_3/imu_annotations.json'
# json_file_path = "/root/shared-nvme/code_result/result/25_01-10/test2/WWADLDatasetSingle_imu_30_3_34_2048_30_0/checkpoint_wifiTAD_34_2048_30_0-epoch-20.pt.json"  # Replace with the actual file path

json_file_path = "/root/shared-nvme/code_result/result/25_01-20/muti_m_t/WWADLDatasetMuti_all_30_3_mamba_layer_8/checkpoint_mamba_mamba_layer_8-epoch-79.pt.json"
# json_file_path = "/root/shared-nvme/code_result/result/25_01-16/single_mamba/WWADLDatasetSingle_all_30_3_mamba_layer_8/checkpoint_mamba_mamba_layer_8-epoch-54.pt.json"

with open(json_file_path, "r") as f:
    predictions = json.load(f)

with open(groundtruth_path, "r") as f:
    groundtruth_json = json.load(f)

raw_predictions1 = predictions['results']['0_2_10.h5']
raw_predictions2
raw_predictions3
raw_predictions4
raw_predictions5
raw_predictions6
groundtruth = groundtruth_json['database']['0_2_10.h5']['annotations']


# 后处理预测结果 较小的值会更严格地去除冗余框，较大的值可能会保留更多框。
processed_predictions = post_process(raw_predictions, conf_thresh=0.5, iou_thresh=0.1, top_k=12)

print(processed_predictions)
save_path = os.path.dirname(json_file_path)
print(save_path)
# 可视化
visualize_results_combined("show", groundtruth, processed_predictions, save_path=save_path)