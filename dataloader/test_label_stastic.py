import os
import numpy as np
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt

seq = '08'
root = '/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences'
label_dir = os.path.join(root, seq, 'labels')
test_label_dir = os.path.join(root, seq, 'test_labels')
pc_dir = os.path.join(root, seq, 'velodyne')
label_mapping_path = 'config/label_mapping/semantic-kitti-sub.yaml'

with open(label_mapping_path, 'r') as f:
    semkittiyaml = yaml.safe_load(f)
learning_map = semkittiyaml['learning_map']

label_files = sorted(os.listdir(label_dir))
test_label_files = sorted(os.listdir(test_label_dir))

for idx, fname in enumerate(label_files):
    if not fname.endswith('.label'):
        continue
    label_path = os.path.join(label_dir, fname)
    test_label_path = os.path.join(test_label_dir, fname)
    pc_path = os.path.join(pc_dir, fname.replace('.label', '.bin'))

    labels = np.fromfile(label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF
    test_labels = np.fromfile(test_label_path, dtype=np.uint32)
    test_sem_labels = test_labels & 0xFFFF

    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)  # [N, 4]

    # 统计当前帧标签分布
    label_count_mapped = defaultdict(int)
    test_label_count_mapped = defaultdict(int)
    for l in sem_labels:
        mapped = learning_map.get(int(l), 0)
        label_count_mapped[int(mapped)] += 1
    for l in test_sem_labels:
        mapped = learning_map.get(int(l), 0)
        test_label_count_mapped[int(mapped)] += 1

    labels_sorted = sorted(set(list(label_count_mapped.keys()) + list(test_label_count_mapped.keys())))
    x = np.arange(len(labels_sorted))
    y_gt = [label_count_mapped.get(k, 0) for k in labels_sorted]
    y_pred = [test_label_count_mapped.get(k, 0) for k in labels_sorted]

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, y_gt, width=0.4, label='GT Label')
    plt.bar(x + 0.2, y_pred, width=0.4, label='Test Label')
    plt.xticks(x, labels_sorted)
    plt.xlabel('Mapped Label')
    plt.ylabel('Count')
    plt.title(f'Frame {idx+1}/{len(label_files)}: {fname} - GT vs Test Label')
    plt.legend()
    plt.tight_layout()
    plt.show()
    input("按回车键显示下一帧...")

print("逐帧可视化结束。")

# print("原始标签统计：")
# for k in sorted(label_count_raw.keys()):
#     print(f"label {k}: {label_count_raw[k]}")

# print("\n映射后标签统计：")
# for k in sorted(label_count_mapped.keys()):
#     print(f"mapped label {k}: {label_count_mapped[k]}")