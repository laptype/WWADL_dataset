import json
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

    Parameters:
    - raw_results: list of dict, raw detection results
    - conf_thresh: float, minimum confidence threshold
    - iou_thresh: float, IoU threshold for NMS
    - top_k: int, maximum number of results to keep
    - use_nms: bool, whether to use NMS

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


def visualize_results_combined(video_name, groundtruth, predictions, title="Action Detection Results"):
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
    plt.show()

# 示例数据
groundtruth = [
    {"segment": [0, 250], "label": "Walking to Bed"},
    {"segment": [250, 650], "label": "Sitting Down"},
    {"segment": [650, 1350], "label": "Reading"},
    {"segment": [1350, 1750], "label": "Pouring Water"},
    {"segment": [1750, 2350], "label": "Taking Medicine"},
    {"segment": [2350, 2750], "label": "Lying Down"},
    {"segment": [2750, 3400], "label": "Using Phone"},
    {"segment": [3400, 3699], "label": "Getting Out of Bed"}
]

json_file_path = "/home/lanbo/WWADL/WWADL_WiFiTAD/output/checkpoint40.json"  # Replace with the actual file path

with open(json_file_path, "r") as f:
    predictions = json.load(f)

raw_predictions = predictions['results']['0_1_14.h5']

# 后处理预测结果
processed_predictions = post_process(raw_predictions, conf_thresh=0.5, iou_thresh=0.3, top_k=10)

print(processed_predictions)

# 可视化
visualize_results_combined("0_1_14.h5", groundtruth, processed_predictions)