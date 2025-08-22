import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data
import pickle
import random
import torch
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

from utils.normalmap import compute_normals_range
from .utils import polarmix
from .transform import Compose

from st_data_help import augment_pose, transform_points

REGISTERED_DATASET_CLASSES = {}
REGISTERED_COLATE_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def register_collate_fn(cls, name=None):
    global REGISTERED_COLATE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_COLATE_CLASSES, f"exist class: {REGISTERED_COLATE_CLASSES}"
    REGISTERED_COLATE_CLASSES[name] = cls
    return cls


def get_dataset_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


def get_collate_class(name):
    global REGISTERED_COLATE_CLASSES
    assert name in REGISTERED_COLATE_CLASSES, f"available class: {REGISTERED_COLATE_CLASSES}"
    return REGISTERED_COLATE_CLASSES[name]


"""
{'car': 0, 'bicycle': 1, 'motorcycle': 2, 'truck': 3, 'other-vehicle': 4, 'person': 5, 'bicyclist': 6, 'motorcyclist': 7, 
'road': 8, 'parking': 9, 'sidewalk': 10, 'other-ground': 11, 'building': 12, 'fence': 13, 'vegetation': 14, 'trunk': 15, 
'terrain': 16, 'pole': 17, 'traffic-sign': 18}
"""
instance_classes = [1, 2, 3, 4, 5, 6, 7,
                    8]  # [2, 3, 4, 5, 7, 8, 10, 13, 14, 17, 20] #
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1)
         * np.pi * 2 / 3]  # x3


@register_dataset
class point_semkitti_mix(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.ignore_label = config['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.instance_aug = loader_config['instance_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.mixing_and_downsampling = loader_config['mix_aug']
        self.polarcutmix = loader_config['polarmix_aug']
        self.bg_trans_aug = loader_config['bg_trans_aug']
        self.bg_dropout_aug = loader_config['bg_dropout_aug']
        self.bg_scale_aug = loader_config['bg_scale_aug']
        self.max_volume_space = config['max_volume_space']
        self.min_volume_space = config['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.rotate_prob = loader_config['rotate_prob']
        self.flip_prob = loader_config['flip_prob']
        self.trans_prob = loader_config['trans_prob']
        self.scale_prob = loader_config['scale_prob']
        self.mix_prob = loader_config['mix_prob']
        self.bg_trans_prob = loader_config['bg_scale_prob']
        self.bg_scale_prob = loader_config['bg_scale_prob']
        self.bg_dropout_prob = loader_config['bg_scale_prob']

        self.polarcutmix_prob = loader_config['polarmix_prob']
        self.seed = loader_config['seed']
        np.random.seed(self.seed)
        print(loader_config)
        self.sec_transform = Compose(cfg=loader_config)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]
        # todo prepare spatio_temporal_data
        data_spatio_temporal_data = self.prepare_spatio_temporal_data(data)  # 增加spatio_temporal_data
        # data_single = self.get_single_sample(data, root, index)
        data_single=self.get_single_sample_with_spatio_temporal(data, root, index,data_spatio_temporal_data)
        # data_single = self.get_single_sample_new(data, root, index)
        # if self.mixing_and_downsampling:
        if self.mixing_and_downsampling and np.random.rand() < self.mix_prob:  # 修改，增加混合概率
            random_integer = np.random.randint(
                low=0, high=len(self.point_cloud_dataset))
            random_index = (
                index+random_integer) % len(self.point_cloud_dataset)
            extra_data, extra_root = self.point_cloud_dataset[random_index]
            extra_spatio_temporal_data = self.prepare_spatio_temporal_data(extra_data)  # 增加spatio_temporal_data
            # extra_single = self.get_single_sample(extra_data, extra_root, random_index, cut_scene=True)
            extra_single = self.get_single_sample_with_spatio_temporal(extra_data, extra_root, random_index, extra_spatio_temporal_data,cut_scene=True)
            # extra_single = self.get_single_sample_new(extra_data, extra_root, random_index, cut_scene=True)
            cutmix_data_dict = {}
            # for keys in data_single.keys():
            #     if keys in ['point_num']:
            #         cutmix_data_dict[keys] = data_single[keys] + \
            #             extra_single[keys]
            #     elif keys == 'ref_index':
            #         extra_single[keys] = extra_single[keys] + \
            #             len(data_single['ref_xyz'])
            #         cutmix_data_dict[keys] = np.concatenate(
            #             (data_single[keys], extra_single[keys]), axis=0)
            #     elif keys in ['ref_xyz', 'ref_label']:
            #         cutmix_data_dict[keys] = np.concatenate(
            #             (data_single[keys], extra_single[keys]), axis=0)
            #     elif keys in ['point_feat', 'point_label', 'normal']:
            #         cutmix_data_dict[keys] = np.concatenate(
            #             (data_single[keys], extra_single[keys]), axis=0)
            #     else:
            #         cutmix_data_dict[keys] = data_single[keys]
            for keys in data_single.keys():
                if keys in ['point_num','st_point_num']:
                    cutmix_data_dict[keys] = data_single[keys] + \
                        extra_single[keys]
                elif keys == 'ref_index':
                    extra_single[keys] = extra_single[keys] + \
                        len(data_single['ref_xyz'])
                    cutmix_data_dict[keys] = np.concatenate(
                        (data_single[keys], extra_single[keys]), axis=0)
                elif keys == 'st_ref_index':
                    extra_single[keys] = extra_single[keys] + \
                        len(data_single['st_ref_xyz'])
                    cutmix_data_dict[keys] = np.concatenate(
                        (data_single[keys], extra_single[keys]), axis=0)
                elif keys in ['ref_xyz', 'ref_label','st_ref_xyz', 'st_ref_label']:
                    cutmix_data_dict[keys] = np.concatenate(
                        (data_single[keys], extra_single[keys]), axis=0)
                elif keys in ['point_feat', 'point_label', 'normal','st_point_feat', 'st_point_label', 'st_normal']:
                    cutmix_data_dict[keys] = np.concatenate(
                        (data_single[keys], extra_single[keys]), axis=0)
                else:
                    cutmix_data_dict[keys] = data_single[keys]

            data_single = self.sec_transform(cutmix_data_dict)
        else:
            data_single = self.sec_transform(data_single)

        return data_single

    def get_single_sample_with_spatio_temporal(self, data, root, index, spatio_temporal_data, cut_scene=False):
        'Generates one sample of data with spatio-temporal data'
        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']
        
        st_xyz=spatio_temporal_data['points'][:,:3]
        st_sig=spatio_temporal_data['points'][:,3]
        st_instance_label=spatio_temporal_data['instance_label']
        st_labels=spatio_temporal_data['labels']
        
        # 单样本预处理（get_single_sample）
        # 空间裁剪：只保留在指定空间范围内的点
        # CutMix/Polarmix：可选的数据混合增强
        # 随机丢点：模拟点云稀疏性
        # 旋转、翻转、缩放、平移：常规点云增强
        # 法向量计算：为每个点计算法向量特征
        # 组装输出：将所有特征、标签、索引等打包成字典
        if self.polarcutmix and np.random.rand() < self.polarcutmix_prob and self.point_cloud_dataset.imageset == 'train':  # 修改，增加 polarmix 概率
            # 如果启用 polarmix 增强，会随机选取另一个样本，将当前样本和另一个样本在极坐标空间内做混合（polarmix），以增强数据多样性。
            random_integer = np.random.randint(
                low=0, high=len(self.point_cloud_dataset))
            extra_data, _ = self.point_cloud_dataset[(
                index+random_integer) % len(self.point_cloud_dataset)]
            # polarmix
            alpha = (np.random.random() - 1) * np.pi
            beta = alpha + np.pi

            # # 这个函数作了两个增强， 一是扇区拼接替换，二是进行了指定的角度旋转
            # xyz, labels = polarmix(xyz, labels, extra_data['xyz'], extra_data['labels'],
            #                        alpha=alpha, beta=beta,
            #                        instance_classes=instance_classes,
            #                        Omega=Omega)
            # 增强所有类别
            xyz, labels, instance_label, sig = polarmix(
                xyz, labels, instance_label, sig,
                extra_data['xyz'], extra_data['labels'], extra_data['instance_label'], extra_data['signal'],
                alpha=alpha, beta=beta,
                instance_classes=instance_classes,
                Omega=Omega,
                target_classes=None
            )
            
        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))
        
        ref_st_pc=st_xyz.copy()
        ref_st_labels=st_labels.copy()
        ref_st_index=np.arange(len(ref_st_pc))
        

        # 布尔值，逻辑且运算
        mask_x = np.logical_and(
            xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(
            xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(
            xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z)) 
        
        st_mask_x = np.logical_and(
            xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        st_mask_y = np.logical_and(
            xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        st_mask_z = np.logical_and(
            xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        st_mask = np.logical_and(st_mask_x, np.logical_and(st_mask_y, st_mask_z))     
        
        if cut_scene:
            mask *= instance_label != 0  # 等价于 mask = mask & (instance_label != 0)，即只有同时满足“在空间范围内”且“实例标签不为0”的点，mask才为True
            st_mask *= st_instance_label != 0  # 等价于 mask = mask & (instance_label != 0)，即只有同时满足“在空间范围内”且“实例标签不为0”的点，mask才为True

        xyz = xyz[mask]
        # ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)
        
        st_xyz=st_xyz[st_mask]
        st_sig=st_sig[st_mask]
        st_instance_label=st_instance_label[st_mask]
        st_point_num=len(st_xyz)
        
        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[
                0]  # xyz.shape[0] 是点数，[0] 是取出满足条件的索引数组
            st_drouput_ratio = np.random.random() * self.max_dropout_ratio
            st_drop_idx = np.where(np.random.random((st_xyz.shape[0])) <= st_drouput_ratio)[0]

            if len(drop_idx) > 0:
                # 这样做不是直接删除点，而是把这些点“挤压”到第一个点的位置，防止点数变化，保证张量形状一致
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]
                
            if len(st_drop_idx) > 0:
                st_xyz[st_drop_idx, :] = st_xyz[0, :]
                st_sig[st_drop_idx, :] = st_sig[0, :] 
                st_instance_label[st_drop_idx] = st_instance_label[0]
            
                    # random data augmentation by flip x , y or x+y
        if self.flip_aug and np.random.rand() < self.flip_prob and self.point_cloud_dataset.imageset == 'train':
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
                st_xyz[:, 0] = -st_xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
                st_xyz[:, 1] = -st_xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
                st_xyz[:, :2] = -st_xyz[:, :2]
                    
        if self.scale_aug and np.random.rand() < self.scale_prob and self.point_cloud_dataset.imageset == 'train':
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
            st_xyz[:, 0] = noise_scale * st_xyz[:, 0]
            st_xyz[:, 1] = noise_scale * st_xyz[:, 1]
        
        if self.transform and np.random.rand() < self.trans_prob and self.point_cloud_dataset.imageset == 'train':
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(
                                            0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            xyz[:, 0:3] += noise_translate
            st_xyz[:, 0:3] += noise_translate
            
        if self.rotate_aug and np.random.rand() < self.rotate_prob and self.point_cloud_dataset.imageset == 'train':
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)
            st_xyz[:, :2] = np.dot(st_xyz[:, :2], j)
            
        feat = np.concatenate((xyz, sig), axis=1)
        unproj_normal_data = compute_normals_range(feat)
            
        st_feat = np.concatenate(st_xyz, st_sig, axis=1)
        st_unproj_normal_data = compute_normals_range(st_feat)
        
        data_dict = {}
        data_dict['point_feat'] = feat  # 对点数据增广后的点云
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['normal'] = unproj_normal_data
        data_dict['root'] = root
        
        data_dict['st_point_feat'] = st_feat
        data_dict['st_point_label'] = st_labels
        data_dict['st_ref_xyz'] = ref_st_pc
        data_dict['st_ref_label'] = ref_st_labels
        data_dict['st_ref_index'] = ref_st_index
        data_dict['st_point_num'] = st_point_num
        data_dict['st_normal'] = st_unproj_normal_data
        
        
        return data_dict

    def get_single_sample(self, data, root, index, cut_scene=False):
        'Generates one sample of data'
        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']

        # 单样本预处理（get_single_sample）
        # 空间裁剪：只保留在指定空间范围内的点。
        # CutMix/Polarmix：可选的数据混合增强。
        # 随机丢点：模拟点云稀疏性。
        # 旋转、翻转、缩放、平移：常规点云增强。
        # 法向量计算：为每个点计算法向量特征。
        # 组装输出：将所有特征、标签、索引等打包成字典。
        if self.polarcutmix and np.random.rand() < self.polarcutmix_prob and self.point_cloud_dataset.imageset == 'train':  # 修改，增加 polarmix 概率
            # 如果启用 polarmix 增强，会随机选取另一个样本，将当前样本和另一个样本在极坐标空间内做混合（polarmix），以增强数据多样性。
            random_integer = np.random.randint(
                low=0, high=len(self.point_cloud_dataset))
            extra_data, _ = self.point_cloud_dataset[(
                index+random_integer) % len(self.point_cloud_dataset)]
            # polarmix
            alpha = (np.random.random() - 1) * np.pi
            beta = alpha + np.pi

            # # 这个函数作了两个增强， 一是扇区拼接替换，二是进行了指定的角度旋转
            # xyz, labels = polarmix(xyz, labels, extra_data['xyz'], extra_data['labels'],
            #                        alpha=alpha, beta=beta,
            #                        instance_classes=instance_classes,
            #                        Omega=Omega)
            
            # 增强所有类别
            xyz, labels, instance_label, sig = polarmix(
                xyz, labels, instance_label, sig,
                extra_data['xyz'], extra_data['labels'], extra_data['instance_label'], extra_data['signal'],
                alpha=alpha, beta=beta,
                instance_classes=instance_classes,
                Omega=Omega,
                target_classes=None
            )
            # # 增强指定类别
            # xyz, labels, instance_label, sig = polarmix(
            #     xyz, labels, instance_label, sig,
            #     extra_data['xyz'], extra_data['labels'], extra_data['instance_label'], extra_data['signal'],
            #     alpha=alpha, beta=beta,
            #     instance_classes=instance_classes,
            #     Omega=Omega,
            #     target_classes=[1,2,3]
            # )

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))
        sig = sig.reshape(-1, 1)

        # 布尔值，逻辑且运算
        mask_x = np.logical_and(
            xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(
            xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(
            xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        if cut_scene:
            mask *= instance_label != 0  # 等价于 mask = mask & (instance_label != 0)，即只有同时满足“在空间范围内”且“实例标签不为0”的点，mask才为True

        xyz = xyz[mask]
        # ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        ### 3D Augmentation ###

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[
                0]  # xyz.shape[0] 是点数，[0] 是取出满足条件的索引数组

            if len(drop_idx) > 0:
                # 这样做不是直接删除点，而是把这些点“挤压”到第一个点的位置，防止点数变化，保证张量形状一致
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        # random data augmentation by flip x , y or x+y
        if self.flip_aug and np.random.rand() < self.flip_prob and self.point_cloud_dataset.imageset == 'train':
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug and np.random.rand() < self.scale_prob and self.point_cloud_dataset.imageset == 'train':
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform and np.random.rand() < self.trans_prob and self.point_cloud_dataset.imageset == 'train':
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(
                                            0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

            # random data augmentation by rotation
        if self.rotate_aug and np.random.rand() < self.rotate_prob and self.point_cloud_dataset.imageset == 'train':
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)
            # 这种做法默认点云的 z 轴是竖直方向，xy 平面是水平面。
            # 也就是说，只有在点云已经对齐到“z轴竖直、xy为地面”的情况下，这种旋转才是合理的。
            #
            # 对于自动驾驶、室外激光点云等，通常采集时就已经保证了这个坐标系（z竖直，xy为地面）。
            # 如果点云不是这种坐标系（比如z不是竖直），那这种旋转就不再是“水平旋转”，可能会导致数据异常。
            # 总结：
            #
            # 这段旋转代码就是二维平面（xy）上的随机旋转（yaw角变换），z不变。
            # 需要点云本身就是“z轴竖直、xy为地面”的坐标系，否则这种增强不合理。

        feat = np.concatenate((xyz, sig), axis=1)

        unproj_normal_data = compute_normals_range(feat)
        
        # 这里可以添加时间通道
        # time_channel = np.zeros((feat.shape[0], 1), dtype=feat.dtype)
        # feat = np.concatenate((feat, time_channel), axis=1)

        data_dict = {}
        data_dict['point_feat'] = feat  # 对点数据增广后的点云
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['normal'] = unproj_normal_data
        data_dict['root'] = root

        return data_dict


    def prepare_spatio_temporal_data(self,data_dict, trans_std=0.1, rot_deg_std=0.5):

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
            
            # 增加时间通道
            time_channel_data=len(data_roots)-idx
            time_channel = np.full((points.shape[0], 1), time_channel_data, dtype=np.float32)
            points = np.concatenate((points, time_channel), axis=1)
            
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

        # 布尔值，逻辑且运算
        mask_x = np.logical_and(
            all_points[:, 0] > self.min_volume_space[0], all_points[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(
            all_points[:, 1] > self.min_volume_space[1], all_points[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(
            all_points[:, 2] > self.min_volume_space[2], all_points[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
        all_points=all_points[mask]
        all_sem_labels = all_sem_labels[mask]
        all_inst_labels = all_inst_labels[mask]
            
        spatio_temporal_data={}
        spatio_temporal_data['points'] = all_points # np.concatenate(all_points, axis=0)
        spatio_temporal_data['labels'] = all_sem_labels # np.concatenate(all_sem_labels, axis=0)
        spatio_temporal_data['instance_label'] = all_inst_labels # np.concatenate(all_inst_labels, axis=0)

        # todo augment merged_points

        return spatio_temporal_data

    def get_single_sample_new(self, data, root, index, cut_scene=False):
        'Generates one sample of data'
        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        if self.polarcutmix and np.random.rand() < self.polarcutmix_prob and self.point_cloud_dataset.imageset == 'train':  # 修改，增加 polarmix 概率
            # 如果启用 polarmix 增强，会随机选取另一个样本，将当前样本和另一个样本在极坐标空间内做混合（polarmix），以增强数据多样性。
            random_integer = np.random.randint(
                low=0, high=len(self.point_cloud_dataset))
            extra_data, _ = self.point_cloud_dataset[(
                index+random_integer) % len(self.point_cloud_dataset)]
            # polarmix
            alpha = (np.random.random() - 1) * np.pi
            beta = alpha + np.pi
            # 这个函数作了两个增强， 一是扇区拼接替换，二是进行了指定的角度旋转
            xyz, labels = polarmix(xyz, labels, extra_data['xyz'], extra_data['labels'],
                                   alpha=alpha, beta=beta,
                                   instance_classes=instance_classes,
                                   Omega=Omega)

           # 空间范围裁剪
        mask_x = np.logical_and(
            xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(
            xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(
            xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        xyz = xyz[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]

        point_num = len(xyz)

        # 1. 分离实例点和背景点
        instance_mask = (instance_label != 0)
        bg_mask = ~instance_mask

        # xyz_instance = xyz[instance_mask]
        # labels_instance = labels[instance_mask]
        # instance_label_instance = instance_label[instance_mask]
        # sig_instance = sig[instance_mask]

        # xyz_bg = xyz[bg_mask]
        # labels_bg = labels[bg_mask]
        # instance_label_bg = instance_label[bg_mask]
        # sig_bg = sig[bg_mask]
        # 防止全为背景或全为实例导致空数组
        if instance_mask.sum() == 0:
            # 全为背景点，实例增强部分跳过
            xyz_instance = np.empty((0, xyz.shape[1]), dtype=xyz.dtype)
            labels_instance = np.empty(
                (0, labels.shape[1]), dtype=labels.dtype)
            instance_label_instance = np.empty(
                (0,), dtype=instance_label.dtype)
            sig_instance = np.empty((0, sig.shape[1]), dtype=sig.dtype)
        else:
            xyz_instance = xyz[instance_mask]
            labels_instance = labels[instance_mask]
            instance_label_instance = instance_label[instance_mask]
            sig_instance = sig[instance_mask]

        if bg_mask.sum() == 0:
            # 全为实例点，背景增强部分跳过
            xyz_bg = np.empty((0, xyz.shape[1]), dtype=xyz.dtype)
            labels_bg = np.empty((0, labels.shape[1]), dtype=labels.dtype)
            instance_label_bg = np.empty((0,), dtype=instance_label.dtype)
            sig_bg = np.empty((0, sig.shape[1]), dtype=sig.dtype)
        else:
            xyz_bg = xyz[bg_mask]
            labels_bg = labels[bg_mask]
            instance_label_bg = instance_label[bg_mask]
            sig_bg = sig[bg_mask]
        # print('labels_bg.shape:', labels_bg.shape)  2 维度
        # print('instance_label_bg.shape:', instance_label_bg.shape) 1 维
        # print('labels_instance.shape:', labels_instance.shape) 2 维
        # print('instance_label_instance.shape:', instance_label_instance.shape) 1 维

        # 2. 只对实例点做增强
        if self.instance_aug and self.point_cloud_dataset.imageset == 'train':
            # 随机丢点（实例点）
            if xyz_instance.shape[0] > 0 and self.dropout and self.point_cloud_dataset.imageset == 'train':
                dropout_ratio = np.random.random() * self.max_dropout_ratio
                drop_idx = np.where(np.random.random(
                    (xyz_instance.shape[0])) <= dropout_ratio)[0]
                if len(drop_idx) > 0:
                    xyz_instance[drop_idx, :] = xyz_instance[0, :]
                    labels_instance[drop_idx, :] = labels_instance[0, :]
                    sig_instance[drop_idx, :] = sig_instance[0, :]
                    instance_label_instance[drop_idx] = instance_label_instance[0]

            if self.flip_aug and np.random.rand() < self.flip_prob:
                flip_type = np.random.choice(4, 1)
                if flip_type == 1:
                    xyz_instance[:, 0] = -xyz_instance[:, 0]
                elif flip_type == 2:
                    xyz_instance[:, 1] = -xyz_instance[:, 1]
                elif flip_type == 3:
                    xyz_instance[:, :2] = -xyz_instance[:, :2]

            if self.scale_aug and np.random.rand() < self.scale_prob:
                noise_scale = np.random.uniform(0.95, 1.05)
                xyz_instance[:, 0] = noise_scale * xyz_instance[:, 0]
                xyz_instance[:, 1] = noise_scale * xyz_instance[:, 1]
                xyz_instance[:, 2] = noise_scale * xyz_instance[:, 2]

            if self.transform and np.random.rand() < self.trans_prob:
                noise_translate = np.array([
                    np.random.normal(0, self.trans_std[0], 1),
                    np.random.normal(0, self.trans_std[1], 1),
                    np.random.normal(0, self.trans_std[2], 1)
                ]).T
                xyz_instance[:, 0:3] += noise_translate

            if self.rotate_aug and np.random.rand() < self.rotate_prob:
                rotate_rad = np.deg2rad(np.random.random() * 360)
                c, s = np.cos(rotate_rad), np.sin(rotate_rad)
                j = np.matrix([[c, s], [-s, c]])
                xyz_instance[:, :2] = np.dot(xyz_instance[:, :2], j)

        # 3. 背景点可选择不做增强，也可做不同增强（如只做平移等，视需求而定）
        # 这里默认不做增强
        if bg_mask.sum() > 0 and self.point_cloud_dataset.imageset == 'train':
            # 随机丢点（背景点）
            if self.bg_dropout_aug and self.point_cloud_dataset.imageset == 'train':
                dropout_ratio = np.random.random() * self.bg_dropout_prob
                drop_idx = np.where(np.random.random(
                    (xyz_bg.shape[0])) <= dropout_ratio)[0]
                if len(drop_idx) > 0:
                    xyz_bg[drop_idx, :] = xyz_bg[0, :]
                    labels_bg[drop_idx, :] = labels_bg[0, :]
                    sig_bg[drop_idx, :] = sig_bg[0, :]
                    instance_label_bg[drop_idx] = instance_label_bg[0]
            # 轻微平移
            if self.bg_trans_aug and np.random.rand() < self.bg_trans_prob:
                noise_translate = np.array([
                    np.random.normal(0, self.trans_std[0], 1),
                    np.random.normal(0, self.trans_std[1], 1),
                    np.random.normal(0, self.trans_std[2], 1)
                ]).T
                xyz_bg[:, 0:3] += noise_translate
            # 轻微缩放
            if self.bg_scale_aug and np.random.rand() < self.bg_scale_prob:
                noise_scale = np.random.uniform(0.98, 1.02)
                xyz_bg[:, 0:3] *= noise_scale

        # 4. 合并
        # 保证标签类为一维
        # labels_instance = labels_instance.reshape(-1)
        # labels_bg = labels_bg.reshape(-1)
        # instance_label_instance = instance_label_instance.reshape(-1)
        # instance_label_bg = instance_label_bg.reshape(-1)

        xyz = np.concatenate([xyz_instance, xyz_bg], axis=0)
        labels = np.concatenate([labels_instance, labels_bg], axis=0)
        # instance_label = np.concatenate([instance_label_instance, instance_label_bg], axis=0)
        sig = np.concatenate([sig_instance, sig_bg], axis=0)

        feat = np.concatenate((xyz, sig), axis=1)
        unproj_normal_data = compute_normals_range(feat)
        
        # 这里可以添加时间通道
        # time_channel = np.zeros((feat.shape[0], 1), dtype=feat.dtype)
        # feat = np.concatenate((feat, time_channel), axis=1)

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc  # useless
        data_dict['ref_label'] = ref_labels  # useless
        data_dict['ref_index'] = ref_index
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['normal'] = unproj_normal_data
        data_dict['root'] = root

        return data_dict


@register_collate_fn
def mix_collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    path = data[0]['root']  # [d['root'] for d in data]

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [torch.from_numpy(d['point_feat']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]
    normal = [torch.from_numpy(d['normal']) for d in data]

    return {
        'points': torch.cat(points).float(),
        'normal': torch.cat(normal).float(),
        'ref_xyz': torch.cat(ref_xyz).float(),  # useless
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(), # useless
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(), # useless
        'path': path,     # useless
        'point_num': point_num, # useless
    }

@register_collate_fn
def mix_collate_fn_with_spatio_temporal_data(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    path = data[0]['root']
    st_point_num = [d['st_point_num'] for d in data]

    b_idx = []
    st_b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
        # st_point_num 需在每个样本的 data_dict 里
        st_b_idx.append(torch.ones(st_point_num[i]) * i)
    points = [torch.from_numpy(d['point_feat']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]
    normal = [torch.from_numpy(d['normal']) for d in data]
    st_points = [torch.from_numpy(d['st_point_feat']) for d in data]
    st_labels = [torch.from_numpy(d['st_point_label']) for d in data]
    st_normal = [torch.from_numpy(d['st_normal']) for d in data]

    return {
        'points': torch.cat(points).float(),
        'normal': torch.cat(normal).float(),
        'ref_xyz': torch.cat(ref_xyz).float(),  # useless
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(), # useless
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),  # useless
        'path': path,   # useless
        'point_num': point_num, # useless
        # 新增st相关
        'st_points': torch.cat(st_points).float(),
        'st_labels': torch.cat(st_labels).long().squeeze(1),
        'st_normal': torch.cat(st_normal).float(),
        'st_batch_idx': torch.cat(st_b_idx).long(),
        'st_point_num': [d['st_point_num'] for d in data],  # useless
    }

@register_dataset
class point_image_dataset_nus(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.max_volume_space = config['max_volume_space']
        self.min_volume_space = config['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        # self.debug = config['debug']

    def __len__(self):
        'Denotes the total number of samples'
        # if self.debug:
        #     return 100 * self.num_vote
        # else:
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(
            xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(
            xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(
            xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        # dropout points
        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random(
                (xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                ref_index[drop_idx] = ref_index[0]

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(
                                            0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        feat = np.concatenate((xyz, sig), axis=1)

        unproj_normal_data = compute_normals_range(feat)

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['normal'] = unproj_normal_data
        data_dict['root'] = root

        return data_dict
