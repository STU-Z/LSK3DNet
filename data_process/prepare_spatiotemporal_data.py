import os
import numpy as np
import yaml
from tqdm import tqdm
import time
def read_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def read_label(label_path):
    return np.fromfile(label_path, dtype=np.uint32)

def read_poses(pose_path):
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('Tr:'):
            Tr = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
            Tr = np.vstack((Tr, [0, 0, 0, 1]))
            return Tr
    raise RuntimeError("Tr not found in calib file.")

def transform_points(points, T):
    xyz1 = np.ones((points.shape[0], 4), dtype=np.float32)
    xyz1[:, :3] = points[:, :3]
    xyz1 = (T @ xyz1.T).T
    points[:, :3] = xyz1[:, :3]
    return points

def get_moving_labels_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    moving_labels = [int(k) for k, v in config['labels'].items() if 'moving' in v]
    return set(moving_labels)

def prepare_spatiotemporal_data(
    lidar_idx,indices, bin_dir, label_dir, pose_path, calib_path, moving_labels
):
    Tr = read_calib(calib_path)
    poses = read_poses(pose_path)
    all_points, all_labels = [], []

    pose_end = poses[lidar_idx]
    pose_end = np.linalg.inv(Tr) @ pose_end @ Tr

    for idx in tqdm(indices, desc="Processing frames"):
        bin_path = os.path.join(bin_dir, f"{idx:06d}.bin")
        label_path = os.path.join(label_dir, f"{idx:06d}.label")
        points = read_bin(bin_path)
        labels = read_label(label_path)

        # 空间范围筛选
        mask = (
            (points[:, 0] >= -50) & (points[:, 0] <= 50) &
            (points[:, 1] >= -50) & (points[:, 1] <= 50) &
            (points[:, 2] >= -4)  & (points[:, 2] <= 2)
        )
        points = points[mask]
        labels = labels[mask]

        # 再排除中心小块
        mask = (
            (np.abs(points[:, 0]) > 2.0) | (np.abs(points[:, 1]) > 2.0)
        )
        points = points[mask]
        labels = labels[mask]

        # 坐标变换
        T_i = np.linalg.inv(Tr) @ poses[idx] @ Tr
        T_end_i = np.linalg.inv(pose_end) @ T_i
        points = transform_points(points, T_end_i)

        all_points.append(points)
        all_labels.append(labels)

    merged_points = np.concatenate(all_points, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    sem_labels = merged_labels & 0xFFFF

    is_moving = np.isin(sem_labels, list(moving_labels))
    is_static = ~is_moving

    return merged_points[is_static], merged_labels[is_static], merged_points[is_moving], merged_labels[is_moving]

if __name__ == "__main__":
    root_dir = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences"
    yaml_path = "/media/zrb/Elements/Git_code/LSK3DNet/config/label_mapping/semantic-kitti-all.yaml"
    # out_root = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/spatiotemporal_out"
    out_root="/home/zrb/temp_cache"

    time_length=3
    moving_labels = get_moving_labels_from_yaml(yaml_path)
    # print(f"Moving labels: {moving_labels}")
    for seq in sorted(os.listdir(root_dir)):
        if seq != "01":
            continue  # 跳过非00序列
        seq_path = os.path.join(root_dir, seq)
        bin_dir = os.path.join(seq_path, "velodyne")
        label_dir = os.path.join(seq_path, "labels")
        pose_path = os.path.join(seq_path, "poses.txt")
        calib_path = os.path.join(seq_path, "calib.txt")
        # out_root = os.path.join(root_dir, seq,"spatiotemporal_out")
 
        if not (os.path.exists(bin_dir) and os.path.exists(label_dir) and os.path.exists(pose_path) and os.path.exists(calib_path)):
            continue

        bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith('.bin')])
        num_frames = len(bin_files)
        if num_frames == 0:
            continue
        for idx in range(0, num_frames):
            # if idx < 1800:
            #     continue
            if idx == 0:
                indices = range(0, 1)
            elif idx < time_length:
                indices = range(0, idx)
            else:
                indices = range(idx-time_length, idx)

            print(f"Processing sequence {seq}, target idx {idx}, using indices {list(indices)}")

            static_points, static_labels, move_points, move_labels = prepare_spatiotemporal_data(
               idx, indices, bin_dir, label_dir, pose_path, calib_path, moving_labels
            )

            # 输出路径，每个目标idx单独保存
            static_bin = os.path.join(out_root, "static", "velodyne", f"{idx:06d}.bin")
            static_label = os.path.join(out_root, "static", "labels", f"{idx:06d}.label")
            move_bin = os.path.join(out_root, "move", "velodyne", f"{idx:06d}.bin")
            move_label = os.path.join(out_root, "move", "labels", f"{idx:06d}.label")

            for p in [static_bin, static_label, move_bin, move_label]:
                os.makedirs(os.path.dirname(p), exist_ok=True)

            max_retry = 3
            for attempt in range(max_retry):
                try:
                    static_points.astype(np.float32).tofile(static_bin)
                    static_labels.astype(np.uint32).tofile(static_label)
                    move_points.astype(np.float32).tofile(move_bin)
                    move_labels.astype(np.uint32).tofile(move_label)
                    break  # 成功写入则跳出循环
                except Exception as e:
                    print(f"写入文件失败: {e}，第{attempt+1}次重试")
                    time.sleep(1)
            else:
                print(f"多次重试后仍写入失败，跳过 idx={idx}")
                continue

            print(f"Saved static: {static_bin}, {static_label}")
            print(f"Saved move: {move_bin}, {move_label}")