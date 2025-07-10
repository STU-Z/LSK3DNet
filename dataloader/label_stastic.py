'''
Author: Zhangrunbang 254616730@qq.com
Date: 2025-06-29 21:27:09
LastEditors: Zhangrunbang 254616730@qq.com
LastEditTime: 2025-06-29 21:27:10
FilePath: /LSK3DNet/dataloader/label_stastic.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import numpy as np
import yaml
from collections import defaultdict

# 路径配置
seq = '08'
root = '/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences'
label_dir = os.path.join(root, seq, 'labels')
label_mapping_path = 'config/label_mapping/semantic-kitti-all.yaml'

# 读取label映射
with open(label_mapping_path, 'r') as f:
    semkittiyaml = yaml.safe_load(f)
learning_map = semkittiyaml['learning_map']

# 统计原始标签
label_count_raw = defaultdict(int)
# 统计映射后标签
label_count_mapped = defaultdict(int)

label_files = sorted(os.listdir(label_dir))
for fname in label_files:
    if not fname.endswith('.label'):
        continue
    label_path = os.path.join(label_dir, fname)
    labels = np.fromfile(label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF  # 原始标签
    for l in sem_labels:
        label_count_raw[int(l)] += 1
        mapped = learning_map.get(int(l), 0)
        label_count_mapped[int(mapped)] += 1

print("原始标签统计：")
for k in sorted(label_count_raw.keys()):
    print(f"label {k}: {label_count_raw[k]}")

print("\n映射后标签统计：")
for k in sorted(label_count_mapped.keys()):
    print(f"mapped label {k}: {label_count_mapped[k]}")