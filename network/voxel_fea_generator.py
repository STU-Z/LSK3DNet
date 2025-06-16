import torch
import torch_scatter
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]

        for idx, scale in enumerate(self.scale_list):
            xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], np.ceil(self.spatial_shape[0] / scale))
            yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale))
            bxyz_indx = torch.stack([data_dict['batch_idx'], xidx, yidx, zidx], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            '''
            1. unq
                含义：bxyz_indx 中所有唯一的行（即所有不同的 [batch_idx, xidx, yidx, zidx] 组合）。
                shape：[M, 4]，M为唯一体素格子的数量。
                作用：每个唯一体素格子的坐标。 
            2. unq_inv
                含义：原始 bxyz_indx 中每个点对应的唯一体素格子的编号（即该点属于 unq 的哪一行）。
                shape：[N]，N为点的数量。
                作用：用于后续聚合操作（如 scatter_mean），把属于同一体素的点聚合到一起。
            3. unq_cnt
                含义：每个唯一体素格子中包含的点的数量。
                shape：[M]，M为唯一体素格子的数量。
                作用：统计每个体素格子里有多少个点。
            '''
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }
        return data_dict


class voxelization_fixvs(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list, voxel_size):
        super(voxelization_fixvs, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz
        self.voxel_size = voxel_size
 
    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]

        for idx, scale in enumerate(self.scale_list):
            xyz_indx = torch.floor(pc / (self.voxel_size * scale))
            xyz_indx -= torch.min(xyz_indx, 0)[0]
            xyz_indx = xyz_indx.long()
            sparse_shape = torch.add(torch.max(xyz_indx, dim=0).values, 1).tolist()[::-1]
            bxyz_indx = torch.stack([data_dict['batch_idx'], xyz_indx[:,0],xyz_indx[:,1],xyz_indx[:,2]], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32),
                'spatial_shape': sparse_shape
            }
        return data_dict
    '''
    bxyz_indx 是形状为 [N, 4] 的张量，每一行是 [batch_idx, xidx, yidx, zidx]，表示每个点属于哪个 batch、哪个体素格子
    unq 所有唯一的体素格子坐标（即所有不同的 [batch_idx, xidx, yidx, zidx]），shape 为 [M, 4]，M为体素格子数。
    unq_inv 每个点对应的唯一体素格子的编号shape 为 [N]。用于后续聚合（如 scatter_mean），可以把属于同一体素的点聚合到一起。
    unq_cnt 每个唯一体素格子中包含的点的数量，shape 为 [M]。
    '''


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(in_channels),

            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
        )

    def prepare_input(self, point, grid_ind, inv_idx, normal=None):
        '''
            作用解释
        torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)

        以 inv_idx 为分组依据，对 point[:, :3]（即每个点的 xyz 坐标）按体素分组求均值。
        输出 shape 是 [体素数, 3]，每一行是一个体素内所有点的 xyz 均值（即该体素的“中心”）。
        [inv_idx]

        用 inv_idx 把每个点映射回它所属体素的均值。
        得到的 pc_mean shape 是 [点数, 3]，每个点都对应它所在体素的中心坐标。

        举例说明
        假设有 6 个点，属于 2 个体素，inv_idx = [0, 0, 1, 1, 1, 0]，则
        
        scatter_mean(..., inv_idx, ...) 得到每个体素的中心
        [inv_idx] 让每个点都能查到自己体素的中心
        '''
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers
        pc_feature = torch.cat((point, nor_pc, center_to_point, normal), dim=1)

        return pc_feature  #  [N, X]

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv'],
            data_dict['normal']
        )
        pt_fea = self.PPmodel(pt_fea) # [N, X] -> [N, out_channels]

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0) #  [M, out_channels]
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(), # # np.int32(self.spatial_shape).tolist()
            batch_size=data_dict['batch_size']
        ) #  [M, out_channels]


        data_dict['coors'] = data_dict['scale_1']['coors'] # [M, 4]，每行是 [batch_idx, zidx, yidx, xidx]
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv'] # [N]，每个点对应的体素编号
        data_dict['full_coors'] = data_dict['scale_1']['full_coors'] # [N, 4]，每行是 [batch_idx, zidx, yidx, xidx]

        return data_dict

class voxel_3d_generator_fixvs(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape, voxel_size):
        super(voxel_3d_generator_fixvs, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.voxel_size = voxel_size
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(in_channels),

            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
        )

    def prepare_input(self, point, grid_ind, inv_idx, normal=None):
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        min_volume_space = torch.floor(torch.min(point[:, :3], 0)[0])
        voxel_centers = grid_ind * self.voxel_size + min_volume_space.to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point, normal), dim=1)

        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv'],
            data_dict['normal']
        )
        pt_fea = self.PPmodel(pt_fea)


        '''
            pt_fea 形状为 [N, C]，N 是点数，C 是特征维度。
            data_dict['scale_1']['coors_inv'] 形状为 [N]，每个点对应的体素编号。
            scatter_mean 以体素编号分组，对每个体素内所有点的特征求均值。
            
            所以，features 的 shape 是 [M, C]
                M：唯一体素的数量（即 coors_inv 的最大值+1）
                C：特征维度
        '''
        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=data_dict['scale_1']['spatial_shape'],
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']
        data_dict['spatial_shape'] = data_dict['scale_1']['spatial_shape']

        return data_dict