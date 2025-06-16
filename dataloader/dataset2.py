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
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8] # [2, 3, 4, 5, 7, 8, 10, 13, 14, 17, 20] #
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3

@register_dataset
class point_semkitti_mix(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.ignore_label = config['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.mixing_and_downsampling = loader_config['mix_aug']
        self.polarcutmix = loader_config['polarmix_aug']
        self.max_volume_space = config['max_volume_space']
        self.min_volume_space = config['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        print(loader_config)
        self.sec_transform = Compose(cfg=loader_config)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)


    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]
        data_single = self.get_single_sample(data, root, index)   
        if self.mixing_and_downsampling:
            random_integer = np.random.randint(low=0, high=len(self.point_cloud_dataset))
            random_index = (index+random_integer)%len(self.point_cloud_dataset)
            extra_data, extra_root = self.point_cloud_dataset[random_index]
            extra_single = self.get_single_sample(extra_data, extra_root, random_index, cut_scene=True) 
            cutmix_data_dict = {}
            for keys in data_single.keys():
                if keys in ['point_num']:
                    cutmix_data_dict[keys] = data_single[keys] + extra_single[keys]
                elif keys == 'ref_index':
                    extra_single[keys] = extra_single[keys] + len(data_single['ref_xyz'])
                    cutmix_data_dict[keys] = np.concatenate((data_single[keys], extra_single[keys]),axis=0)
                elif keys in ['ref_xyz', 'ref_label']:
                    cutmix_data_dict[keys] = np.concatenate((data_single[keys], extra_single[keys]),axis=0)
                elif keys in ['point_feat', 'point_label', 'normal']:
                    cutmix_data_dict[keys] = np.concatenate((data_single[keys], extra_single[keys]),axis=0)
                else:
                    cutmix_data_dict[keys] = data_single[keys]

            data_single = self.sec_transform(cutmix_data_dict)                    
        else:
            data_single = self.sec_transform(data_single)

        return data_single

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
        if self.polarcutmix:
            #如果启用 polarmix 增强，会随机选取另一个样本，将当前样本和另一个样本在极坐标空间内做混合（polarmix），以增强数据多样性。
            random_integer = np.random.randint(low=0, high=len(self.point_cloud_dataset))
            extra_data, _ = self.point_cloud_dataset[(index+random_integer)%len(self.point_cloud_dataset)]
            # polarmix
            alpha = (np.random.random() - 1) * np.pi
            beta = alpha + np.pi

            # 这个函数作了两个增强， 一是扇区拼接替换，二是进行了指定的角度旋转
            xyz, labels = polarmix(xyz, labels, extra_data['xyz'], extra_data['labels'],
                                      alpha=alpha, beta=beta,
                                      instance_classes=instance_classes,
                                      Omega=Omega)
            
        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))


        # 布尔值，逻辑且运算
        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
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

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]  # xyz.shape[0] 是点数，[0] 是取出满足条件的索引数组

            if len(drop_idx) > 0:
                # 这样做不是直接删除点，而是把这些点“挤压”到第一个点的位置，防止点数变化，保证张量形状一致
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]


        ### 3D Augmentation ###
        # random data augmentation by rotation
        if self.rotate_aug:
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
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        feat = np.concatenate((xyz, sig), axis=1)

        unproj_normal_data = compute_normals_range(feat)

        data_dict = {}
        data_dict['point_feat'] = feat  #  对点数据增广后的点云
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
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
    path = data[0]['root'] # [d['root'] for d in data]

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
        'ref_xyz': torch.cat(ref_xyz).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(),
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),
        'path': path,
        'point_num': point_num, 
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

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
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
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

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
                                        np.random.normal(0, self.trans_std[1], 1),
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
