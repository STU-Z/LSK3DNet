import numpy as np
import open3d as o3d
import sys
import yaml
import os
def load_color_map_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    color_map = config['color_map']
    # 转为标签最大值+1的数组，未定义的label用[0,0,0]
    max_label = max(color_map.keys())
    color_arr = np.zeros((max_label + 1, 3), dtype=np.float32)
    for k, v in color_map.items():
        color_arr[int(k)] = np.array(v[::-1]) / 255.0  # BGR转RGB并归一化
    return color_arr

def read_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def read_label(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF
    return sem_labels

def visualize_points(points, sem_labels=None, color_arr=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    print(f"点云点数: {points.shape[0]}")
    if sem_labels is not None and color_arr is not None:
        sem_labels = np.clip(sem_labels, 0, color_arr.shape[0] - 1)
        colors = color_arr[sem_labels]
    else:
        intensity = points[:, 3]
        intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-8)
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = intensity
        colors[:, 1] = 1 - intensity
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # if len(sys.argv) < 4:
    #     print("用法: python visual_semantickitti_bin.py <bin文件路径> <label文件路径> <yaml配置路径>")
    #     sys.exit(1)
    # bin_path = sys.argv[1]
    # label_path = sys.argv[2]
    # yaml_path = sys.argv[3]
    if len(sys.argv) < 2:
       print("用法: python visual_semantickitti_bin.py <000000> ")
       sys.exit(1)
    lidar_idx=int(sys.argv[1])
    bin_path ="/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/spatiotemporal_out/move/velodyne/"
    label_path ="/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/spatiotemporal_out/move/labels"
    yaml_path = "config/label_mapping/semantic-kitti-all.yaml"
    bin_path = os.path.join(bin_path, f"{lidar_idx:06d}.bin")
    label_path = os.path.join(label_path, f"{lidar_idx:06d}.label")
    print(f"正在可视化点云: {bin_path}")
    print(f"对应标签: {label_path}")
    points = read_bin(bin_path)
    sem_labels = read_label(label_path)
    color_arr = load_color_map_from_yaml(yaml_path)
    if sem_labels.shape[0] != points.shape[0]:
        print("警告: label 点数与点云点数不一致，无法可视化语义标签。")
        sem_labels = None
    visualize_points(points, sem_labels, color_arr)