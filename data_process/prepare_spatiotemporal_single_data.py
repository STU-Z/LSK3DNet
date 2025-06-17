'''
    准备时空数据
    1. 读取每一个索引对应的.bin文件和.label文件
    2. 读取标定文件、读取每一帧的位姿，按照位姿将读取的索引数据转换到当前坐标系下
    3. 将转换后的点云数据和对应标签数据合并转换为numpy数组
    4. 将数据分别存储为.bin和.label文件
'''
import os
import numpy as np
import yaml
from tqdm import tqdm
import time
def read_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def read_label(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels

def read_poses(pose_path):
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))  # 4x4
            poses.append(T)
    return poses

def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            P[key] = np.array([float(x) for x in value.strip().split()]).reshape(-1)
    Tr = P['Tr'].reshape(3, 4)
    Tr = np.vstack((Tr, [0, 0, 0, 1]))
    return Tr

def transform_points(points, T):
    # points: [N, 3] or [N, 4], T: [4, 4]
    xyz1 = np.ones((points.shape[0], 4), dtype=np.float32)
    xyz1[:, :3] = points[:, :3]
    # xyz1 = xyz1 @ T.T
    xyz1 =(T @ xyz1.T).T
    points[:, :3] = xyz1[:, :3]
    return points

# def prepare_spatiotemporal_data(
#     lidar_idx,indices, bin_dir, label_dir, pose_path, calib_path, out_bin, out_label
# ):
#     # 读取标定和位姿
#     Tr = read_calib(calib_path)
#     poses = read_poses(pose_path)

#     print(f"Tr: {Tr}")

#     all_points = []
#     all_labels = []

#     print(f"len(indices): {len(indices)}")
#     pose_end=poses[lidar_idx]
#     pose_end = np.linalg.inv(Tr) @ pose_end @ Tr
#     for idx in tqdm(indices, desc="Processing frames"):
#         bin_path = os.path.join(bin_dir, f"{idx:06d}.bin")
#         label_path = os.path.join(label_dir, f"{idx:06d}.label")
#         points = read_bin(bin_path)
#         labels = read_label(label_path)

#         # 只保留在指定范围内的点
#         mask = (
#             (points[:, 0] >= -50) & (points[:, 0] <= 50) &
#             (points[:, 1] >= -50) & (points[:, 1] <= 50) &
#             (points[:, 2] >= -4)  & (points[:, 2] <= 2)
#         )
#         points = points[mask]
#         labels = labels[mask]

#         # print('poses[idx]: ', poses[idx])
#         # 坐标变换到当前帧
#         T_i = np.linalg.inv(Tr) @ poses[idx] @ Tr
#         T_end_i=np.linalg.inv(pose_end) @ T_i

#         # points = transform_points(points, T_i)
#         points = transform_points(points, T_end_i)

#         all_points.append(points)
#         all_labels.append(labels)
    
#     print(f"T(idx): {idx,poses[idx]}")

#     # 合并
#     merged_points = np.concatenate(all_points, axis=0)
#     merged_labels = np.concatenate(all_labels, axis=0)

#     # 保存
#     merged_points.astype(np.float32).tofile(out_bin)
#     merged_labels.astype(np.uint32).tofile(out_label)
#     print(f"Saved merged bin: {out_bin}")
#     print(f"Saved merged label: {out_label}")

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

def get_moving_labels_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    moving_labels = [int(k) for k, v in config['labels'].items() if 'moving' in v]
    return set(moving_labels)

if __name__ == "__main__":
    # 示例参数（请根据实际路径修改）
    lidar_idx=50
    indices = range(lidar_idx-5, lidar_idx)  # 处理前5帧
    out_root= "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/spatiotemporal_out"
    bin_dir = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/velodyne"
    label_dir = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/labels"
    pose_path = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/poses.txt"
    calib_path = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/calib.txt"
    out_bin = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/st_data/velodyne/merged.bin"
    out_label = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/st_data/labels/merged.label"
    yaml_path = "/media/zrb/Elements/Git_code/LSK3DNet/config/label_mapping/semantic-kitti-all.yaml"
    moving_labels = get_moving_labels_from_yaml(yaml_path)
    static_points, static_labels, move_points, move_labels = prepare_spatiotemporal_data(
               lidar_idx, indices, bin_dir, label_dir, pose_path, calib_path, moving_labels
            )
        # 输出路径，每个目标idx单独保存
    static_bin = os.path.join(out_root, "static", "velodyne", f"{lidar_idx:06d}.bin")
    static_label = os.path.join(out_root, "static", "labels", f"{lidar_idx:06d}.label")
    move_bin = os.path.join(out_root, "move", "velodyne", f"{lidar_idx:06d}.bin")
    move_label = os.path.join(out_root, "move", "labels", f"{lidar_idx:06d}.label")
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
        print(f"多次重试后仍写入失败，跳过 idx={lidar_idx}")   