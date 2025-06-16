import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, large_kernel, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        '''
        卷积核（kernel size）不会影响 M，M 只由稀疏体素的位置决定。

        详细解释
        在 spconv 的稀疏卷积（如 SubMConv3d）中，M 表示稀疏体素的数量，即非零体素的个数。
        卷积核大小（kernel size）只影响每个体素特征的计算方式，不会改变稀疏体素的分布和数量。
        只有在下采样（如 SparseConv3d 带 stride > 1 或 pool 操作）时，才会导致 M 变小（体素数减少）。
        举例
        输入 [M, in_channels]，经过 SubMConv3d（无下采样），输出是 [M, out_channels]，M 不变。
        如果用 SparseConv3d 且 stride > 1，则输出的 M 会变小。
        '''
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=large_kernel,indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=large_kernel, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        详细解释
            identity = self.layers_in(x)

            self.layers_in 是一个 1x1 的稀疏卷积 + BN，作用类似于残差结构中的“shortcut”分支。
            这里得到的 identity 是输入 x 经过简单线性变换后的特征。
            output = self.layers(x)

            self.layers 是一系列大卷积核的稀疏卷积、BN 和激活，提取更复杂的特征。
            output.replace_feature(...)

            output.features 是主分支的输出特征，identity.features 是 shortcut 分支的特征。
            两者相加后，经过 LeakyReLU 激活，作为新的特征，替换到稀疏张量 output 里。
            replace_feature 是 spconv 的方法，用于用新的特征替换稀疏张量的 feature 字段。
        '''
        identity = self.layers_in(x) # [M, in_channels] --> [M, out_channels]
        output = self.layers(x) # [M, in_channels] --> [M, out_channels]
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1)) #  [M, out_channels]


class point_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(point_encoder, self).__init__()
        self.scale = scale
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            # out_channels // 2 就是把 out_channels 除以2，结果为整数。
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        # coors [M,4] , p_fea [M,C]
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale  # 整除 scale 前后，inv 可能会变化，因为体素分辨率变粗，多个原体素可能合并为一个新体素
        inv = torch.unique(torch.cat([batch, coors], dim=1), return_inverse=True, dim=0)[1]
        '''
        详细解释:
            coors = coors[:, 1:] // scale 这一步是对体素坐标进行下采样（变粗），即把原本细分的体素合并成更大的体素格子。
            torch.unique(torch.cat([batch, coors], dim=1), return_inverse=True, dim=0)[1] 会根据下采样后的体素坐标重新分组，生成新的 inv。
        举例说明:
            假设原始体素坐标有： [0, 2, 4, 6]
            如果 scale=2，整除后变为： [0, 1, 2, 3]
            如果有多个体素整除后落在同一个格子，比如 [0, 1, 2, 3, 4] // 2 = [0, 0, 1, 1, 2]
            那么原本属于不同体素的点，整除后会被分到同一个体素，inv 也会变成一样。
        结论:
            整除 scale 前后，inv 可能会变化，因为体素分辨率变粗，多个原体素可能合并为一个新体素。
            只有当 scale=1 时，整除操作不影响体素分组，inv 不变。
        总结：
            体素坐标整除 scale 后，inv 通常会变化，因为体素分辨率变粗，分组方式发生了变化。
        '''
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict):
        output, inv = self.downsample(data_dict['coors'], features) # [M, C] [M, C] --> [M, C] [M]
        identity = self.layer_in(features) # [M, C] --> [M, outchannels]
        '''
        output = self.PPmodel(output)[inv]
        作用解释
            self.PPmodel(output) 的输出 shape 是 [M, C]，M 为体素数，每个体素有一个特征。
            inv 是 shape 为 [N] 的索引数组，N 为点数，inv[i] 表示第 i 个点属于第几个体素。
            [inv] 的作用是：

            把每个体素的特征“还原”到每个点上，即让每个点都获得它所属体素的特征。
            结果 shape 是 [N, C]，每个点都查到自己体素的特征。
            举例
            假设：

            有 3 个体素，体素特征为 [[a], [b], [c]]
            有 5 个点，inv = [0, 2, 1, 0, 2]
            则 self.PPmodel(output)[inv] 得到 [a, c, b, a, c]
            一句话总结：
            [inv] 的作用是把体素特征映射回每个点，让每个点都获得它所属体素的特征。
        '''
        output = self.PPmodel(output)[inv] # [M, C] --> [M, outchannels]

        output = torch.cat([identity, output], dim=1) # [M, outchannels] + [M, outchannels] --> [M, 2*outchannels]

        '''
        output[data_dict['coors_inv']]:
            output 是 shape [M, C] 的张量，M 为体素数，C 为特征维度。:
            data_dict['coors_inv'] 是 shape [N] 的索引数组，N 为点数，每个点对应它所属体素的编号（范围 0~M-1）。
        这句代码的作用是：
            把每个体素的特征“映射”回每个点，让每个点都获得它所属体素的特征
        举例：
            假设：
                有 3 个体素，体素特征为 output = [[a], [b], [c]]
                有 5 个点，data_dict['coors_inv'] = [0, 2, 1, 0, 2]
            则 output[data_dict['coors_inv']] 获取到的结果为 [a, c, b, a, c]
            [M,C] --> [N,C]
        '''
        v_feat = torch_scatter.scatter_mean(
            self.layer_out(output[data_dict['coors_inv']]),
            data_dict['scale_{}'.format(self.scale)]['coors_inv'],
            dim=0
        ) # [N, C] --> [M, outchannels]
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']
        
        return v_feat #[M, outchannels]

class point_encoder_fixvs(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(point_encoder_fixvs, self).__init__()
        self.scale = scale
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        inv = torch.unique(torch.cat([batch, coors], 1), return_inverse=True, dim=0)[1]
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict):
        output, inv = self.downsample(data_dict['coors'], features)
        identity = self.layer_in(features)
        output = self.PPmodel(output)[inv]
        output = torch.cat([identity, output], dim=1)

        v_feat = torch_scatter.scatter_mean(
            self.layer_out(output[data_dict['coors_inv']]),
            data_dict['scale_{}'.format(self.scale)]['coors_inv'],
            dim=0
        )
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']
        data_dict['spatial_shape'] = data_dict['scale_{}'.format(self.scale)]['spatial_shape']

        return v_feat

class SPVBlock(nn.Module):
    def __init__(self, large_kernel, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(large_kernel, in_channels, out_channels, self.indice_key),
            SparseBasicBlock(large_kernel, out_channels, out_channels, self.indice_key),
        )
        self.p_enc = point_encoder(in_channels, out_channels, scale)

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']  # [N]
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv'] # [N]

        # voxel encoder
        v_fea = self.v_enc(data_dict['sparse_tensor']) # [M, C]，M为体素数，C为特征维度
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features # [M, C]，当前尺度体素的特征
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors'] #  [N, 4]
        '''
        coors_inv_last 和 coors_inv 的区别
            coors_inv_last：上一级（上一个尺度）体素的点到体素的映射索引，长度为点数，值域为上一级体素数。
            coors_inv：当前尺度体素的点到体素的映射索引，长度为点数，值域为当前体素数。
            2. v_fea.features[coors_inv_last] 的意义
                v_fea.features 是当前体素的特征（shape [M, C]），但这里用的是 [coors_inv_last]，即把上一级体素的特征映射到每个点。
                这样，v_fea.features[coors_inv_last] 得到的是每个点继承自上一级体素的特征。
            3. torch_scatter.scatter_mean(..., coors_inv, dim=0)
                以当前尺度的体素索引为分组，把所有点（每个点带着上一级体素特征）聚合到当前体素上，求均值。
                这样实现了跨尺度的信息融合：当前体素的特征不仅包含本层的信息，还融合了上一级体素的信息。
        一句话总结：
            用 coors_inv_last 是为了实现跨尺度（跨层）特征融合，把上一级体素的信息传递到当前体素；只用 coors_inv 就无法实现多尺度信息的有效流动。
        
            举例：
            v_fea.features = [
                [1.0, 1.1],   # 当前体素0的特征
                [2.0, 2.2],   # 当前体素1的特征
            ]
            # shape: [2, 2]

            coors_inv_last = [0, 1, 1, 0, 1]
            
            v_fea.features[coors_inv_last] 得到：
            [
              [1.0, 1.1],  # 点1，对应体素0
              [2.0, 2.2],  # 点2，对应体素1
              [2.0, 2.2],  # 点3，对应体素1
              [1.0, 1.1],  # 点4，对应体素0
              [2.0, 2.2],  # 点5，对应体素1
            ]
        '''

        '''
        v_fea.features[coors_inv_last] 的 shape 是 [N, C]，N 为点数，C 为特征维度。
        coors_inv 的 shape 是 [N]，每个点对应当前尺度体素的编号。
        torch_scatter.scatter_mean(..., coors_inv, dim=0) 会按体素编号分组，把所有属于同一体素的点特征求均值，输出 shape 为 [M, C]，M 为当前尺度体素数
        '''
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0) # [M, C]

        # point encoder
        p_fea = self.p_enc(
            features=data_dict['sparse_tensor'].features+v_fea.features, # [M,C]
            data_dict=data_dict
        )   # [M, C]，

        # fusion and pooling
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=p_fea+v_fea_inv, # [M, C] + [M, C] = [M, C]
            indices=data_dict['coors'], # [M, 4]  [batch_idx, z, y, x]
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )

        return p_fea[coors_inv]  # [M,C] --> [N,C]


class SPVBlock_fixvs(nn.Module):
    def __init__(self, large_kernel, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape):
        super(SPVBlock_fixvs, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(large_kernel, in_channels, out_channels, self.indice_key),
            SparseBasicBlock(large_kernel, out_channels, out_channels, self.indice_key),
        )
        self.p_enc = point_encoder_fixvs(in_channels, out_channels, scale)

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']

        # voxel encoder
        v_fea = self.v_enc(data_dict['sparse_tensor'])
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors']
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)

        # point encoder
        p_fea = self.p_enc(
            features=data_dict['sparse_tensor'].features+v_fea.features,
            data_dict=data_dict
        )

        # fusion and pooling
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=p_fea+v_fea_inv,
            indices=data_dict['coors'],
            spatial_shape=data_dict['spatial_shape'],
            batch_size=data_dict['batch_size']
        )

        return p_fea[coors_inv]

