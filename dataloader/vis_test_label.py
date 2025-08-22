import os
import numpy as np
import yaml
import open3d as o3d

import matplotlib.pyplot as plt

seq = '08'
root = '/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences'
label_dir = os.path.join(root, seq, 'labels')
test_label_dir = os.path.join(root, seq, 'test_labels')
pc_dir = os.path.join(root, seq, 'velodyne')
label_mapping_path = 'config/label_mapping/semantic-kitti-sub.yaml'

def load_color_arr_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    color_map = config['color_map']
    max_label = max([int(k) for k in color_map.keys()])
    color_arr = np.zeros((max_label + 1, 3), dtype=np.float32)
    for k, v in color_map.items():
        color_arr[int(k)] = np.array(v[::-1]) / 255.0  # BGR转RGB并归一化
    return color_arr

color_arr = load_color_arr_from_yaml(label_mapping_path)

with open(label_mapping_path, 'r') as f:
    semkittiyaml = yaml.safe_load(f)
learning_map = semkittiyaml['learning_map']

label_files = sorted(os.listdir(label_dir))
test_label_files = sorted(os.listdir(test_label_dir))

target_label = 32  # 你要统计的类别（可修改）

for idx, fname in enumerate(label_files):
    if not fname.endswith('.label'):
        continue
    label_path = os.path.join(label_dir, fname)
    test_label_path = os.path.join(test_label_dir, fname)
    pc_path = os.path.join(pc_dir, fname.replace('.label', '.bin'))
    print("Processing file:", label_path)
    labels = np.fromfile(label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF
    test_labels = np.fromfile(test_label_path, dtype=np.uint32)
    test_sem_labels = test_labels & 0xFFFF

    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # [N, 3]

    # GT标签点云
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points)
    # colors_gt = color_arr[np.clip(mapped_labels, 0, color_arr.shape[0] - 1)]
    colors_gt = color_arr[sem_labels]
    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)

    # Test标签点云
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points + np.array([150, 0, 0]))
    colors_pred = color_arr[np.clip(test_sem_labels, 0, color_arr.shape[0] - 1)]
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)


    # 统计指定类别被预测为哪些类别（以真值为参考）
    mask = sem_labels == target_label
    # # 用 learning_map 映射 test 标签
    # mapped_test_labels = np.array([learning_map.get(int(l), 0) for l in test_sem_labels])
    pred_labels_for_target = test_sem_labels[mask]
    unique, counts = np.unique(pred_labels_for_target, return_counts=True)
    total = counts.sum()
    print(f"GT类别 {target_label} 在测试标签中被统计为:")
    for u, c in zip(unique, counts):
        percent = c / total * 100 if total > 0 else 0
        print(f"  {u}: {c} 次，占比 {percent:.2f}%")
    
    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_pred],
        window_name=f'Frame {idx+1}/{len(label_files)}: GT(left) vs Test(right) - {fname}',
        width=1600, height=800,
        point_show_normal=False
    )

print("逐帧点云标签可视化结束。")