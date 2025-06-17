import numpy as np
import open3d as o3d
import sys
import yaml
import os
import glob

def load_color_map_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    color_map = config['color_map']
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

def visualize_points(points, sem_labels=None, color_arr=None, window_name="SemanticKITTI Viewer"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
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
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.get_render_option().background_color = np.array([1, 1, 1])  # 白色背景
    return vis, pcd

def main(bin_dir, label_dir, yaml_path):
    # 获取所有bin文件，按文件名排序
    bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
    label_files = [os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + ".label") for f in bin_files]
    color_arr = load_color_map_from_yaml(yaml_path)
    idx = [0]  # 用列表包裹以便在回调中修改

    def load_and_update(vis, pcd, idx):
        points = read_bin(bin_files[idx[0]])
        sem_labels = read_label(label_files[idx[0]])
        print(f"点云数量: {points.shape[0]}")  # 新增：输出点数
        if sem_labels.shape[0] != points.shape[0]:
            print(f"警告: {bin_files[idx[0]]} label 点数与点云点数不一致，无法可视化语义标签。")
            sem_labels = None
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if sem_labels is not None:
            sem_labels = np.clip(sem_labels, 0, color_arr.shape[0] - 1)
            colors = color_arr[sem_labels]
        else:
            intensity = points[:, 3]
            intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-8)
            colors = np.zeros((points.shape[0], 3))
            colors[:, 0] = intensity
            colors[:, 1] = 1 - intensity
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.get_view_control().set_zoom(0.45)
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.get_render_option().show_coordinate_frame = False
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color

    def next_frame(vis):
        if idx[0] < len(bin_files) - 1:
            idx[0] += 1
            load_and_update(vis, pcd, idx)
            print(f"显示: {os.path.basename(bin_files[idx[0]])}")
        return False

    def prev_frame(vis):
        if idx[0] > 0:
            idx[0] -= 1
            load_and_update(vis, pcd, idx)
            print(f"显示: {os.path.basename(bin_files[idx[0]])}")
        return False

    def quit_viewer(vis):
        vis.close()
        return True

    # 初始显示
    points = read_bin(bin_files[idx[0]])
    sem_labels = read_label(label_files[idx[0]])
    print(f"点云数量: {points.shape[0]}")  # 新增：输出点数
    vis, pcd = visualize_points(points, sem_labels, color_arr)
    print(f"显示: {os.path.basename(bin_files[idx[0]])}")


    # 注册按键回调
    vis.register_key_callback(ord("N"), lambda vis: next_frame(vis))
    vis.register_key_callback(ord("P"), lambda vis: prev_frame(vis))
    vis.register_key_callback(ord("Q"), lambda vis: quit_viewer(vis))
    print("按 N 下一帧，P 上一帧，Q 退出")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python visual_semantickitti_bin.py <bin目录> <label目录> <yaml配置路径>")
        sys.exit(1)
    bin_dir = sys.argv[1]
    label_dir = sys.argv[2]
    yaml_path = sys.argv[3]
    main(bin_dir, label_dir, yaml_path)