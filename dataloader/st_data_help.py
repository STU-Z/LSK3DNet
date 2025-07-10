import os
import numpy as np

from scipy.spatial.transform import Rotation as R
from utils.normalmap import compute_normals_range
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


def load_semkitti_bin_with_spatio_temporal(bin_path, learning_map, imageset='train', length=None):
    """
    输入bin文件路径，返回data_dict
    """
    dir_path = os.path.dirname(bin_path)
    # print(f"dir_path: {dir_path}")

    raw_data = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    xyz, feat = raw_data[:, :3], raw_data[:, 3:4]
    origin_len = len(raw_data)
    calib_path = os.path.join(os.path.dirname(
        os.path.dirname(bin_path)), 'calib.txt')
    # print(f"calib_path: {calib_path}")
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
    # print(f"current_lidar_idx: {current_lidar_idx}")
    poses_dir_path = dir_path.replace('velodyne', 'poses_split')
    # print(f"poses_dir_path: {poses_dir_path}")
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

    spatio_temporal_data = {}
    spatio_temporal_data['data_paths'] = spatio_temporal_data_paths  # to get 'xyz' and 'signal'
    spatio_temporal_data['poses'] = spatio_temporal_poses
    spatio_temporal_data['calib_pose'] = calib_data
    spatio_temporal_data['labels_paths'] = spatio_temporal_labels_paths
    spatio_temporal_data['current_pose']=np.load(os.path.join(poses_dir_path, f"{current_lidar_idx:06d}.npy"))
    
    data_dict = {}
    # current frame info
    data_dict['xyz'] = xyz
    data_dict['labels'] = sem_data.astype(np.uint8)
    data_dict['instance_label'] = inst_data
    data_dict['signal'] = feat
    data_dict['origin_len'] = origin_len
    
    # spatio temporal info
    data_dict['spatio_temporal_data'] = spatio_temporal_data

    return data_dict

def transform_points(points, T):
    xyz1 = np.ones((points.shape[0], 4), dtype=np.float32)
    xyz1[:, :3] = points[:, :3]
    xyz1 = (T @ xyz1.T).T
    points[:, :3] = xyz1[:, :3]
    return points



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
    
# def spatio_temporal_augment(data, config, cut_scene=False):
#         'Generates one sample of data'
#         xyz = data['xyz']
       
#         sig = data['signal']


#         # 单样本预处理（get_single_sample）
#         # 空间裁剪：只保留在指定空间范围内的点。
#         # CutMix/Polarmix：可选的数据混合增强。
#         # 随机丢点：模拟点云稀疏性。
#         # 旋转、翻转、缩放、平移：常规点云增强。
#         # 法向量计算：为每个点计算法向量特征。
#         # 组装输出：将所有特征、标签、索引等打包成字典。
#         # if self.polarcutmix and np.random.rand() < self.polarcutmix_prob and self.point_cloud_dataset.imageset == 'train':  # 修改，增加 polarmix 概率
#         #     #如果启用 polarmix 增强，会随机选取另一个样本，将当前样本和另一个样本在极坐标空间内做混合（polarmix），以增强数据多样性。
#         #     random_integer = np.random.randint(low=0, high=len(self.point_cloud_dataset))
#         #     extra_data, _ = self.point_cloud_dataset[(index+random_integer)%len(self.point_cloud_dataset)]
#         #     # polarmix
#         #     alpha = (np.random.random() - 1) * np.pi
#         #     beta = alpha + np.pi

#         #     # 这个函数作了两个增强， 一是扇区拼接替换，二是进行了指定的角度旋转
#         #     xyz, labels = polarmix(xyz, labels, extra_data['xyz'], extra_data['labels'],
#         #                               alpha=alpha, beta=beta,
#         #                               instance_classes=instance_classes,
#         #                               Omega=Omega)
            
#         ref_pc = xyz.copy()
#         # ref_labels = labels.copy()
#         ref_index = np.arange(len(ref_pc))


#         # 布尔值，逻辑且运算
#         mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
#         mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
#         mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
#         mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

#         # if cut_scene:
#         #     mask *= instance_label != 0  # 等价于 mask = mask & (instance_label != 0)，即只有同时满足“在空间范围内”且“实例标签不为0”的点，mask才为True
            
#         xyz = xyz[mask]
#         # ref_pc = ref_pc[mask]
#         # labels = labels[mask]
#         # instance_label = instance_label[mask]
#         ref_index = ref_index[mask]
#         sig = sig[mask]
#         point_num = len(xyz)

#         ### 3D Augmentation ###
               
#         if config['dropout_aug'] and config['imageset'] == 'train':
#             dropout_ratio = np.random.random() * config['max_dropout_ratio']
#             drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]  # xyz.shape[0] 是点数，[0] 是取出满足条件的索引数组

#             if len(drop_idx) > 0:
#                 # 这样做不是直接删除点，而是把这些点“挤压”到第一个点的位置，防止点数变化，保证张量形状一致
#                 xyz[drop_idx, :] = xyz[0, :]
#                 # labels[drop_idx, :] = labels[0, :]
#                 sig[drop_idx, :] = sig[0, :]
#                 # instance_label[drop_idx] = instance_label[0]
#                 ref_index[drop_idx] = ref_index[0]




#         # random data augmentation by flip x , y or x+y
#         if config['flip_aug'] and np.random.rand() < config['flip_prob'] and config['imageset'] == 'train':
#             flip_type = np.random.choice(4, 1)
#             if flip_type == 1:
#                 xyz[:, 0] = -xyz[:, 0]
#             elif flip_type == 2:
#                 xyz[:, 1] = -xyz[:, 1]
#             elif flip_type == 3:
#                 xyz[:, :2] = -xyz[:, :2]

#         if config['scale_aug'] and np.random.rand() < config['scale_prob'] and config['imageset'] == 'train':
#             noise_scale = np.random.uniform(0.95, 1.05)
#             xyz[:, 0] = noise_scale * xyz[:, 0]
#             xyz[:, 1] = noise_scale * xyz[:, 1]

#         if config['transform'] and np.random.rand() < config['trans_prob'] and config['imageset'] == 'train':
#             noise_translate = np.array([np.random.normal(0, 0.1, 1),
#                                         np.random.normal(0, 0.1, 1),
#                                         np.random.normal(0, 0.1, 1)]).T

#             xyz[:, 0:3] += noise_translate
            
#                # random data augmentation by rotation
#         if config['rotate_aug'] and np.random.rand() < config['rotate_prob'] and config['imageset'] == 'train':
#             rotate_rad = np.deg2rad(np.random.random() * 360)
#             c, s = np.cos(rotate_rad), np.sin(rotate_rad)
#             j = np.matrix([[c, s], [-s, c]])
#             xyz[:, :2] = np.dot(xyz[:, :2], j)
#             # 这种做法默认点云的 z 轴是竖直方向，xy 平面是水平面。
#             # 也就是说，只有在点云已经对齐到“z轴竖直、xy为地面”的情况下，这种旋转才是合理的。
#             # 
#             # 对于自动驾驶、室外激光点云等，通常采集时就已经保证了这个坐标系（z竖直，xy为地面）。
#             # 如果点云不是这种坐标系（比如z不是竖直），那这种旋转就不再是“水平旋转”，可能会导致数据异常。
#             # 总结：
#             # 
#             # 这段旋转代码就是二维平面（xy）上的随机旋转（yaw角变换），z不变。
#             # 需要点云本身就是“z轴竖直、xy为地面”的坐标系，否则这种增强不合理。


#         feat = np.concatenate((xyz, sig), axis=1)

#         unproj_normal_data = compute_normals_range(feat)

#         data_dict = {}
#         data_dict['st_point_feat'] = feat  #  对点数据增广后的点云
#         # data_dict['point_label'] = labels
#         data_dict['st_ref_xyz'] = ref_pc
#         # data_dict['ref_label'] = ref_labels
#         data_dict['st_ref_index'] = ref_index
#         data_dict['st_point_num'] = point_num
#         # data_dict['origin_len'] = origin_len
#         data_dict['normal'] = unproj_normal_data
#         # data_dict['root'] = root
        
#         return data_dict
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
        # # 空间范围筛选
        
        # mask = (
        #     (points[:, 0] >= -50) & (points[:, 0] <= 50) &
        #     (points[:, 1] >= -50) & (points[:, 1] <= 50) &
        #     (points[:, 2] >= -4) & (points[:, 2] <= 2)
        # )
        # points = points[mask]
        # labels = labels[mask]
        # # 再排除中心小块
        # mask = (
        #     (np.abs(points[:, 0]) > 2.0) | (np.abs(points[:, 1]) > 2.0)
        # )
        # points = points[mask]
        # labels = labels[mask]
        
        # 分离语义标签和实例标签
        sem_labels = (labels & 0xFFFF).astype(np.uint16)
        inst_labels = (labels >> 16).astype(np.uint16)
        # 坐标变换
        T_i = np.linalg.inv(Tr) @ poses[idx] @ Tr
        T_end_i = np.linalg.inv(pose_end) @ T_i
        T_end_i_noise=augment_pose(T_end_i, trans_std, rot_deg_std)
        points = transform_points(points, T_end_i_noise)
        
        # 拼接 points 和 feat
        points_feat = np.concatenate([points, feat], axis=1)
        
        
        all_points.append(points_feat)
        all_sem_labels.append(sem_labels)
        all_inst_labels.append(inst_labels)
    
    data_dict['spatio_temporal_data']['points'] = np.concatenate(all_points, axis=0)
    data_dict['spatio_temporal_data']['sem_labels'] = np.concatenate(all_sem_labels, axis=0)
    data_dict['spatio_temporal_data']['inst_labels'] = np.concatenate(all_inst_labels, axis=0)
    
    # todo augment merged_points

    return data_dict
