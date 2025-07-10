import os
import numpy as np
from torch.utils import data
import yaml
import pickle
from pathlib import Path
from nuscenes.utils import splits


import open3d as o3d

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train', label_mapping="waymo.yaml", num_vote=1):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.num_vote = num_vote
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths_vote(
                '/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        # 说明，self.im_idx[index]形为：path/00/velodyne/000000.bin
        # 其中，path是数据集的根目录，00是序列号，
        # velodyne是点云数据所在的文件夹，000000.bin是点

        raw_data = np.fromfile(
            self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        xyz, feat = raw_data[:, :3], raw_data[:, 3:4]
        origin_len = len(raw_data)

        if self.imageset == 'test':
            sem_data = np.expand_dims(np.zeros_like(
                raw_data[:, 0], dtype=int), axis=1)
            inst_data = np.expand_dims(np.zeros_like(
                raw_data[:, 0], dtype=np.uint32), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))

            sem_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data >> 16

            # annotated_data 是从 .label 文件读出来的，每个点一个 32 位无符号整数。
            # 这 32 位里，低 16 位是语义标签，高 16 位是实例标签。
            # & 0xFFFF 作用是只保留低 16 位，即语义标签部分。

        origin_len = len(xyz)

        data_dict = {}
        data_dict['xyz'] = xyz
        data_dict['labels'] = sem_data.astype(np.uint8)
        data_dict['instance_label'] = inst_data
        data_dict['signal'] = feat
        data_dict['origin_len'] = origin_len

        return data_dict, self.im_idx[index]


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def absoluteFilePaths_vote(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label

# load Semantic KITTI class info


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map']
                            [i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('Tr:'):
            Tr = np.array([float(x)
                          for x in line.strip().split()[1:]]).reshape(3, 4)
            Tr = np.vstack((Tr, [0, 0, 0, 1]))
            return Tr
    raise RuntimeError("Tr not found in calib file.")


def load_semkitti_bin(bin_path, learning_map, imageset='train', length=None):
    """
    输入bin文件路径，返回data_dict
    """
    dir_path = os.path.dirname(bin_path)
    print(f"dir_path: {dir_path}")

    raw_data = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    xyz, feat = raw_data[:, :3], raw_data[:, 3:4]
    origin_len = len(raw_data)
    calib_path = os.path.join(os.path.dirname(
        os.path.dirname(bin_path)), 'calib.txt')
    print(f"calib_path: {calib_path}")
    Tr = read_calib(calib_path)
    # poses = read_poses(pose_path)

    if imageset == 'test':
        sem_data = np.expand_dims(np.zeros_like(
            raw_data[:, 0], dtype=int), axis=1)
        inst_data = np.expand_dims(np.zeros_like(
            raw_data[:, 0], dtype=np.uint32), axis=1)
    else:
        label_path = bin_path.replace('velodyne', 'labels')[:-3] + 'label'
        annotated_data = np.fromfile(
            label_path, dtype=np.uint32).reshape((-1, 1))
        sem_data = annotated_data & 0xFFFF
        sem_data = np.vectorize(learning_map.__getitem__)(sem_data)
        inst_data = annotated_data >> 16

    current_lidar_idx = int(os.path.splitext(os.path.basename(bin_path))[0])
    print(f"current_lidar_idx: {current_lidar_idx}")
    poses_dir_path = dir_path.replace('velodyne', 'poses_split')
    print(f"poses_dir_path: {poses_dir_path}")
    label_dir_path = dir_path.replace('velodyne', 'labels')

    spatio_temporal_data_paths = []
    spatio_temporal_poses_paths = []

    spatio_temporal_poses = []
    calib_data = []
    if current_lidar_idx < length:
        if current_lidar_idx == 0:
            idx_range = range(0, current_lidar_idx+1)
        else:
            idx_range = range(
                max(0, current_lidar_idx-length), current_lidar_idx)
    else:
        idx_range = range(current_lidar_idx-length, current_lidar_idx)
    spatio_temporal_data_paths = [os.path.join(
        dir_path, f"{idx:06d}.bin") for idx in idx_range]
    spatio_temporal_labels_paths = [os.path.join(
        label_dir_path, f"{idx:06d}.label") for idx in idx_range]
    spatio_temporal_poses_paths = [os.path.join(
        poses_dir_path, f"{idx:06d}.npy") for idx in idx_range]
    spatio_temporal_poses = [np.load(pose_file)
                             for pose_file in spatio_temporal_poses_paths]
    calib_data = [Tr for _ in range(len(spatio_temporal_poses))]
    print(f"len(spatio_temporal_poses): {len(spatio_temporal_poses)}")
    print(f"spatio_temporal_poses: {spatio_temporal_poses}")
    print(
        f"len(spatio_temporal_data_paths): {len(spatio_temporal_data_paths)}")
    print(f"spatio_temporal_data_paths: {spatio_temporal_data_paths}")
    print(f"len(calib_data): {len(calib_data)}")
    # print(f"bin_paths: {idx_st_data_paths}")
    # print(f"idx_poses_paths: {idx_poses_paths}")
    # print(f"len(idx_st_data_paths): {len(idx_st_data_paths)}")
    # print(f"len(idx_poses_paths): {len(idx_poses_paths)}")
    # spatio_temporal_data_paths.append(idx_st_data_paths)
    # spatio_temporal_poses_paths.append(idx_poses_paths)
    # print(f"spatio_temporal_data_paths: {spatio_temporal_data_paths}")
    # print(f"spatio_temporal_poses_paths: {spatio_temporal_poses_paths}")
    # print(f"len(spatio_temporal_data_paths): {len(spatio_temporal_data_paths)}")
    # print(f"len(spatio_temporal_poses_paths): {len(spatio_temporal_poses_paths)}")
    spatio_temporal_data = {}
    spatio_temporal_data['data_paths'] = spatio_temporal_data_paths
    spatio_temporal_data['poses'] = spatio_temporal_poses
    spatio_temporal_data['calib_pose'] = calib_data
    spatio_temporal_data['labels_paths'] = spatio_temporal_labels_paths
    spatio_temporal_data['current_pose']=np.load(os.path.join(poses_dir_path, f"{current_lidar_idx:06d}.npy"))
    
    data_dict = {}
    data_dict['xyz'] = xyz
    data_dict['labels'] = sem_data.astype(np.uint8)
    data_dict['instance_label'] = inst_data
    data_dict['signal'] = feat
    data_dict['origin_len'] = origin_len
    data_dict['spatio_temporal_data'] = spatio_temporal_data

    return data_dict

def transform_points(points, T):
    xyz1 = np.ones((points.shape[0], 4), dtype=np.float32)
    xyz1[:, :3] = points[:, :3]
    xyz1 = (T @ xyz1.T).T
    points[:, :3] = xyz1[:, :3]
    return points

from scipy.spatial.transform import Rotation as R

def augment_pose(pose, trans_std=0.1, rot_deg_std=2.0):
    """
    pose: 4x4 numpy array
    trans_std: 平移扰动标准差（米）
    rot_deg_std: 旋转扰动标准差（度）
    """
    # 平移增广
    noise_t = np.random.normal(0, trans_std, size=(3,))
    pose_aug = pose.copy()
    pose_aug[:3, 3] += noise_t

    # 旋转增广（绕z轴为例）
    noise_angle = np.random.normal(0, rot_deg_std)
    rot_noise = R.from_euler('z', noise_angle, degrees=True).as_matrix()
    pose_aug[:3, :3] = rot_noise @ pose_aug[:3, :3]
    return pose_aug
    # data_dict['spatio_temporal_data']['poses'] = [
    # augment_pose(pose) for pose in data_dict['spatio_temporal_data']['poses']
    # ]   
def gather_spatio_temporal_data(data_dict):

    data_roots = data_dict['spatio_temporal_data']['data_paths']
    labels_paths = data_dict['spatio_temporal_data']['labels_paths']
    poses = data_dict['spatio_temporal_data']['poses']
    calib_poses = data_dict['spatio_temporal_data']['calib_pose']

    # 检查长度是否一致
    assert len(data_roots) == len(
        labels_paths), f"data_paths和labels_paths长度不一致: {len(data_roots)} vs {len(labels_paths)}"
    assert len(data_roots) == len(
        poses), f"data_paths和poses长度不一致: {len(data_roots)} vs {len(poses)}"
    assert len(data_roots) == len(
        calib_poses), f"data_paths和calib_pose长度不一致: {len(data_roots)} vs {len(calib_poses)}"
    pose_end=data_dict['spatio_temporal_data']['current_pose']
    pose_end = np.linalg.inv(calib_poses[0]) @ pose_end @ calib_poses[0]
    
    all_points, all_sem_labels,all_inst_labels = [], [], []
    for idx in range(len(data_roots)):
        bin_path = data_roots[idx]
        # pose = poses[idx]
        Tr = calib_poses[idx]
        labels = np.fromfile(labels_paths[idx], dtype=np.uint32)
        # 读取bin文件
        raw_data = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
        points, feat = raw_data[:, :3], raw_data[:, 3:4]
        # 空间范围筛选
        mask = (
            (points[:, 0] >= -50) & (points[:, 0] <= 50) &
            (points[:, 1] >= -50) & (points[:, 1] <= 50) &
            (points[:, 2] >= -4) & (points[:, 2] <= 2)
        )
        points = points[mask]
        labels = labels[mask]
        # 再排除中心小块
        mask = (
            (np.abs(points[:, 0]) > 2.0) | (np.abs(points[:, 1]) > 2.0)
        )
        points = points[mask]
        labels = labels[mask]
        # 分离语义标签和实例标签
        sem_labels = (labels & 0xFFFF).astype(np.uint16)
        inst_labels = (labels >> 16).astype(np.uint16)
        # 坐标变换
        T_i = np.linalg.inv(Tr) @ poses[idx] @ Tr
        T_end_i = np.linalg.inv(pose_end) @ T_i
        points = transform_points(points, T_end_i)
        
        all_points.append(points)
        all_sem_labels.append(sem_labels)
        all_inst_labels.append(inst_labels)
    
    merged_points = np.concatenate(all_points, axis=0)
    merged_sem_labels = np.concatenate(all_sem_labels, axis=0)
    merged_inst_labels = np.concatenate(all_inst_labels, axis=0)

    return merged_points, merged_sem_labels, merged_inst_labels

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


def prepare_spatio_temporal_data(data_dict, trans_std=0.1, rot_deg_std=0.5):

        data_roots = data_dict['spatio_temporal_data']['data_paths']
        labels_paths = data_dict['spatio_temporal_data']['labels_paths']
        poses = data_dict['spatio_temporal_data']['poses']
        calib_poses = data_dict['spatio_temporal_data']['calib_pose']

        # 检查长度是否一致
        assert len(data_roots) == len(
            labels_paths), f"data_paths和labels_paths长度不一致: {len(data_roots)} vs {len(labels_paths)}"
        assert len(data_roots) == len(
            poses), f"data_paths和poses长度不一致: {len(data_roots)} vs {len(poses)}"
        assert len(data_roots) == len(
            calib_poses), f"data_paths和calib_pose长度不一致: {len(data_roots)} vs {len(calib_poses)}"
        pose_end = data_dict['spatio_temporal_data']['current_pose']
        pose_end = np.linalg.inv(calib_poses[0]) @ pose_end @ calib_poses[0]

        all_points, all_sem_labels, all_inst_labels = [], [], []
        for idx in range(len(data_roots)):
            bin_path = data_roots[idx]
            # pose = poses[idx]
            Tr = calib_poses[idx]

            labels = np.fromfile(labels_paths[idx], dtype=np.uint32)
            # 读取bin文件
            raw_data = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
            points, feat = raw_data[:, :3], raw_data[:, 3:4]
            # # 空间范围筛选

            # mask = (
            #     (points[:, 0] >= -50) & (points[:, 0] <= 50) &
            #     (points[:, 1] >= -50) & (points[:, 1] <= 50) &
            #     (points[:, 2] >= -4) & (points[:, 2] <= 2)
            # )
            # points = points[mask]
            # labels = labels[mask]
            # feat=feat[mask]
            # # 再排除中心小块
            # mask = (
            #     (np.abs(points[:, 0]) > 2.0) | (np.abs(points[:, 1]) > 2.0)
            # )
            # points = points[mask]
            # labels = labels[mask]
            # feat=feat[mask]
            # print(f"points shape: {points.shape}, labels shape: {labels.shape}, feat shape: {feat.shape}")
            # 分离语义标签和实例标签
            sem_labels = (labels & 0xFFFF).astype(np.uint16)
            inst_labels = (labels >> 16).astype(np.uint16)
            # 坐标变换
            T_i = np.linalg.inv(Tr) @ poses[idx] @ Tr
            T_end_i = np.linalg.inv(pose_end) @ T_i
            T_end_i_noise = augment_pose(T_end_i, trans_std, rot_deg_std)
            points = transform_points(points, T_end_i_noise)

            # 拼接 points 和 feat
            points_feat = np.concatenate([points, feat], axis=1)
            all_points.append(points_feat)
            all_sem_labels.append(sem_labels)
            all_inst_labels.append(inst_labels)
        
        all_points = np.concatenate(all_points, axis=0)
        all_sem_labels = np.concatenate(all_sem_labels, axis=0)
        all_inst_labels = np.concatenate(all_inst_labels, axis=0)

        mask_points = (
                (all_points[:, 0] >= -50) & (all_points[:, 0] <= 50) &
                (all_points[:, 1] >= -50) & (all_points[:, 1] <= 50) &
                (all_points[:, 2] >= -4) & (all_points[:, 2] <= 2)
            )
        mask_points_near= (
                (np.abs(all_points[:, 0]) > 3.0) | (np.abs(all_points[:, 1]) > 3.0)
            )
        # mask = np.logical_and(mask_points, mask_points_near)
        all_points=all_points[mask_points]
        all_sem_labels = all_sem_labels[mask_points]
        all_inst_labels = all_inst_labels[mask_points]
            
        spatio_temporal_data={}
        spatio_temporal_data['points'] = all_points # np.concatenate(all_points, axis=0)
        spatio_temporal_data['sem_labels'] = all_sem_labels # np.concatenate(all_sem_labels, axis=0)
        spatio_temporal_data['instance_label'] = all_inst_labels # np.concatenate(all_inst_labels, axis=0)

        # todo augment merged_points

        return spatio_temporal_data
# 生成实例颜色（每个实例一个随机色，0为背景黑色）
def get_instance_color_map(instance_labels):
    unique_ids = np.unique(instance_labels)
    color_map = {}
    np.random.seed(42)
    for uid in unique_ids:
        if uid == 0:
            color_map[uid] = np.array([0, 0, 0])  # 背景黑
        else:
            color_map[uid] = np.random.rand(3)
    return color_map

if __name__ == "__main__":
    # 指定bin文件路径和label_mapping路径
    bin_path = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences/00/velodyne/001565.bin"
    label_mapping = "config/label_mapping/semantic-kitti-all.yaml"
    yaml_path = "config/label_mapping/semantic-kitti-all.yaml"
    # 读取learning_map
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml['learning_map']

    # 加载数据
    data_dict = load_semkitti_bin(
        bin_path, learning_map, imageset='train', length=5)

    # merged_points, merged_labels, merged_instance_labels = gather_spatio_temporal_data(data_dict)
    st_data=prepare_spatio_temporal_data(data_dict)
    merged_points= st_data['points']
    merged_labels = st_data['sem_labels']
    merged_instance_labels = st_data['instance_label']
    
    color_arr = load_color_map_from_yaml(yaml_path)
    if merged_labels.shape[0] != merged_points.shape[0]:
        print("警告: label 点数与点云点数不一致，无法可视化语义标签。")
        merged_labels = None
    # num_instance_points = np.sum(merged_instance_labels != 0)
    # print(f"merged_instance_labels中不为0的点数: {num_instance_points}")
    
    # mask_x = np.logical_and(merged_points[:, 0] > -50, merged_points[:, 0] < 50)
    # mask_y = np.logical_and(merged_points[:, 1] > -50, merged_points[:, 1] < 50)
    # mask_z = np.logical_and(merged_points[:, 2] > -4, merged_points[:, 2] < 4)
    # mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
    # mask *= merged_instance_labels != 0
    # instance_points = merged_points[mask]
    # instance_sem_labels= merged_labels[mask] if merged_labels is not None else None
    # instance_labels = merged_instance_labels[mask]
    
    # 1. merged_points 可视化
    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(merged_points[:, :3])
    if merged_labels is not None and color_arr is not None:
        merged_labels = np.clip(merged_labels, 0, color_arr.shape[0] - 1)
        colors_merged = color_arr[merged_labels]
    else:
        intensity = merged_points[:, 3] if merged_points.shape[1] > 3 else np.zeros(merged_points.shape[0])
        intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-8)
        colors_merged = np.zeros((merged_points.shape[0], 3))
        colors_merged[:, 0] = intensity
        colors_merged[:, 1] = 1 - intensity
    pcd_merged.colors = o3d.utility.Vector3dVector(colors_merged)


    
    # 2. xyz 可视化（纯黑色）
    xyz = data_dict['xyz']
    pcd_xyz = o3d.geometry.PointCloud()
    pcd_xyz.points = o3d.utility.Vector3dVector(xyz)
    colors_xyz = np.zeros((xyz.shape[0], 3))  # 全黑
    pcd_xyz.colors = o3d.utility.Vector3dVector(colors_xyz)

    # 3. 同时显示
    # o3d.visualization.draw_geometries([pcd_merged, pcd_xyz])     
    # # 3. 同时显示，并分别设置点大小
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Merged & XYZ", width=800, height=600)
    vis.add_geometry(pcd_merged)
    # vis.add_geometry(pcd_xyz)

    render_option = vis.get_render_option()
    render_option.point_size = 3.0  # 设置全局点大小

    # 分别设置点大小（Open3D 0.17+ 支持 per-geometry 点大小）
    try:
        # pcd_merged.paint_uniform_color([1, 1, 1])  # 先设置为白色，实际颜色已在上面赋值
        pcd_merged.point["size"] = np.full((len(pcd_merged.points),), 3.0)  # 3.0为merged_points点大小
        pcd_xyz.point["size"] = np.full((len(pcd_xyz.points),), 6.0)        # 6.0为xyz点大小
    except Exception as e:
        print("Open3D 版本不支持 per-geometry 点大小，仅支持全局点大小。")

    vis.run()
    vis.destroy_window() 
    
      