import os
import json
import matplotlib.pyplot as plt

# IoU 计算函数
def iou(segment1, segment2):
    start1, end1 = segment1
    start2, end2 = segment2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0

# NMS 函数
def nms(results, iou_thresh=0.5):
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    final_results = []
    while sorted_results:
        best = sorted_results.pop(0)
        final_results.append(best)
        sorted_results = [
            result for result in sorted_results
            if iou(best['segment'], result['segment']) < iou_thresh
        ]
    return final_results

# 后处理函数
def post_process(raw_results, conf_thresh=0.5, iou_thresh=0.1, top_k=12):
    filtered_results = [res for res in raw_results if res['score'] >= conf_thresh]
    filtered_results = nms(filtered_results, iou_thresh=iou_thresh)
    filtered_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)[:top_k]
    return filtered_results

# 可视化函数
def visualize_results_combined(video_name, groundtruth, predictions, save_path=None):
    all_labels = set([gt['label'] for gt in groundtruth] + [pred['label'] for pred in predictions])
    color_map = {label: plt.cm.tab20(i / len(all_labels)) for i, label in enumerate(all_labels)}
    plt.figure(figsize=(15, 6))

    for gt in groundtruth:
        start, end = gt['segment']
        label = gt['label']
        plt.plot([start, end], [1, 1], color=color_map[label], linewidth=8, label=f"GT: {label}")

    for pred in predictions:
        start, end = pred['segment']
        label = pred['label']
        score = pred['score']
        plt.plot([start, end], [0, 0], color=color_map[label], linewidth=8, label=f"Pred: {label} ({score:.2f})")

    plt.title(f"Results for {video_name}", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.yticks([0, 1], ["Predictions", "Groundtruth"], fontsize=12)
    plt.ylim(-1, 2)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f'{video_name}.jpg'))
        print(f"Saved visualization to {save_path}/{video_name}.jpg")
    else:
        plt.show()
    plt.close()

# 转换时间为秒并排序
def convert_and_sort(results, sampling_rate=50):
    for video_name in results:
        # 将 groundtruth 和 predictions 中的时间段转为秒并保留两位小数
        for gt in results[video_name]['groundtruth']:
            gt['segment'] = [round(t / sampling_rate, 2) for t in gt['segment']]
        for pred in results[video_name]['predictions']:
            pred['segment'] = [round(t / sampling_rate, 2) for t in pred['segment']]
            pred['score'] = round(pred['score'], 2)  # 将 score 保留两位小数
        # 按开始时间排序
        results[video_name]['groundtruth'] = sorted(results[video_name]['groundtruth'], key=lambda x: x['segment'][0])
        results[video_name]['predictions'] = sorted(results[video_name]['predictions'], key=lambda x: x['segment'][0])
    return results

# 主脚本
def main():
    # 设置路径
    groundtruth_path = '/root/shared-nvme/dataset/all_30_3/imu_annotations.json'
    json_file_path = "/root/shared-nvme/code_result/result/25_01-20/muti_m_t/WWADLDatasetMuti_all_30_3_mamba_layer_8/checkpoint_mamba_mamba_layer_8-epoch-79.pt.json"

    # 加载数据
    with open(json_file_path, "r") as f:
        predictions = json.load(f)

    with open(groundtruth_path, "r") as f:
        groundtruth_json = json.load(f)

    # 输出保存路径
    save_path = os.path.dirname(json_file_path)
    os.makedirs(save_path, exist_ok=True)
    visualization_path = os.path.join(save_path, "visualizations")
    os.makedirs(visualization_path, exist_ok=True)

    # 结果字典
    results = {}

    # 遍历所有样本
    for video_name, raw_predictions in predictions['results'].items():
        groundtruth = groundtruth_json['database'][video_name]['annotations']
        processed_predictions = post_process(raw_predictions, conf_thresh=0.5, iou_thresh=0.1, top_k=12)
        results[video_name] = {
            "groundtruth": groundtruth,
            "predictions": processed_predictions
        }
        # 可视化并保存图片
        visualize_results_combined(video_name, groundtruth, processed_predictions, save_path=visualization_path)

    # 转换时间为秒并排序
    results = convert_and_sort(results, sampling_rate=50)

    # 保存 JSON 文件
    output_file_path = os.path.join(save_path, "processed_results.json")
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Processed results saved to {output_file_path}")

# 执行脚本
if __name__ == "__main__":
    main()
