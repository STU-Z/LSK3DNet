'''
Author: Zhangrunbang 254616730@qq.com
Date: 2025-08-21 20:03:14
LastEditors: Zhangrunbang 254616730@qq.com
LastEditTime: 2025-08-21 20:18:51
FilePath: /LSK3DNet/code_test/test_factorized_submconv3d.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch import SubMConv3d
from spconv.pytorch.conv import SparseConvolution
from typing import Optional, Union, List, Tuple


class FactorizedWeightSubMConv3d(SubMConv3d):
    """
    基于因子分解的权重矩阵子流形卷积层
    
    权重矩阵M[i,j,k]由三个一维数组a[0], a[1], a[2]生成，其中M[i,j,k] = a[0][i] * a[1][j] * a[2][k]
    
    权重矩阵参数参与训练: factors_requires_grad: bool = True
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]] = 1,
                 padding: Union[int, List[int]] = 0,
                 dilation: Union[int, List[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 weight_factors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                 factors_requires_grad: bool = True,** kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs
        )
        
        # 处理kernel_size为列表形式
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 3
        else:
            assert len(kernel_size) == 3, "kernel_size must be 3D for 3D convolution"
            self.kernel_size = kernel_size
        
        k_d, k_h, k_w = self.kernel_size
        
        # 初始化三个维度的因子数组
        self.weight_factors = self._initialize_weight_factors(weight_factors, k_d, k_h, k_w)
        
        # 设置是否可学习
        for factor in self.weight_factors:
            factor.requires_grad = factors_requires_grad
    
    def _initialize_weight_factors(self, weight_factors, k_d, k_h, k_w):
        """初始化三个维度的因子数组"""
        if weight_factors is not None:
            # 检查输入的因子是否符合要求
            assert len(weight_factors) == 3, "weight_factors must be a tuple of 3 tensors"
            assert weight_factors[0].shape == (k_d,), f"First factor must have length {k_d}"
            assert weight_factors[1].shape == (k_h,), f"Second factor must have length {k_h}"
            assert weight_factors[2].shape == (k_w,), f"Third factor must have length {k_w}"
            
            return nn.ParameterList([torch.nn.Parameter(factor.clone()) for factor in weight_factors])
        else:
            # 随机初始化因子（更易产生梯度）
            return nn.ParameterList([
                torch.nn.Parameter(torch.randn(k_d)),
                torch.nn.Parameter(torch.randn(k_h)),
                torch.nn.Parameter(torch.randn(k_w))
            ])
    
    def _generate_weight_matrix(self):
        """根据三个因子数组生成权重矩阵 M[i,j,k] = a[0][i] * a[1][j] * a[2][k]"""
        a_d, a_h, a_w = self.weight_factors
        # 使用广播机制计算三维权重矩阵
        weight_matrix = a_d.view(-1, 1, 1) * a_h.view(1, -1, 1) * a_w.view(1, 1, -1)
        return weight_matrix
    
    # def forward(self, input: spconv.SparseConvTensor):
    #     # 1. 生成因子权重矩阵
    #     weight_matrix = self._generate_weight_matrix()  # 形状: (k_d, k_h, k_w)
        
    #     # 2. 扩展因子矩阵形状，与原始卷积核相乘
    #     weight_matrix_expanded = weight_matrix.view(1, 1, *self.kernel_size)  # [1, 1, k_d, k_h, k_w]
    #     weighted_weight = self.weight * weight_matrix_expanded  # 逐元素相乘
        
    #     # 3. 保存原始权重，使用新权重进行计算，之后恢复原始权重
    #     original_weight = self.weight.data
    #     self.weight.data = weighted_weight.data  # 临时替换权重
        
    #     # 调用父类的forward方法执行卷积
    #     output = super().forward(input)
        
    #     # 恢复原始权重
    #     self.weight.data = original_weight
        
    #     return output
    
    def forward(self, input: spconv.SparseConvTensor):
        # 1. 生成因子权重矩阵并扩展维度
        weight_matrix = self._generate_weight_matrix()  # (k_d, k_h, k_w)
        weight_matrix_expanded = weight_matrix.view(1, 1, *self.kernel_size)  # [1,1,k_d,k_h,k_w]
        
        # 2. 计算最终卷积权重（原始权重 × 因子矩阵）
        weighted_weight = self.weight * weight_matrix_expanded  # 保留计算图
        
        # 3. 调用父类的 forward 方法，传入加权后的权重
        # 这里需要用 super().forward，并临时替换 self.weight
        original_weight = self.weight.data.clone()
        self.weight.data = weighted_weight.data
        output = super().forward(input)
        self.weight.data = original_weight  # 恢复原始权重
        return output
    
    def get_weight_factors(self):
        """获取三个维度的因子数组"""
        return self.weight_factors
    
    def set_weight_factors(self, new_factors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """手动设置三个维度的因子数组"""
        k_d, k_h, k_w = self.kernel_size
        assert len(new_factors) == 3, "new_factors must be a tuple of 3 tensors"
        assert new_factors[0].shape == (k_d,), f"First factor must have length {k_d}"
        assert new_factors[1].shape == (k_h,), f"Second factor must have length {k_h}"
        assert new_factors[2].shape == (k_w,), f"Third factor must have length {k_w}"
        
        with torch.no_grad():
            self.weight_factors[0].copy_(new_factors[0])
            self.weight_factors[1].copy_(new_factors[1])
            self.weight_factors[2].copy_(new_factors[2])


def point_cloud_to_voxel(points, voxel_size, coors_range):
    """手动将点云转换为体素数据（替代PointToVoxel）"""
    vsize_x, vsize_y, vsize_z = voxel_size
    x_min, y_min, z_min, x_max, y_max, z_max = coors_range
    
    # 提取坐标和特征
    coordinates = points[:, :3]  # (N, 3)
    features = points[:, 3:]     # (N, C)
    num_features = features.shape[1]
    
    # 过滤超出坐标范围的点
    mask = (coordinates[:, 0] >= x_min) & (coordinates[:, 0] < x_max) & \
           (coordinates[:, 1] >= y_min) & (coordinates[:, 1] < y_max) & \
           (coordinates[:, 2] >= z_min) & (coordinates[:, 2] < z_max)
    coordinates = coordinates[mask]
    features = features[mask]
    if len(coordinates) == 0:
        return np.zeros((0, num_features), dtype=np.float32), np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.int32)
    
    # 计算体素索引
    voxel_indices = np.floor((coordinates - [x_min, y_min, z_min]) / [vsize_x, vsize_y, vsize_z]).astype(np.int32)
    
    # 为体素索引添加批次维度（默认为0）
    batch_indices = np.zeros(len(voxel_indices), dtype=np.int32)
    coors_with_batch = np.column_stack([batch_indices, voxel_indices])  # (N, 4)：(batch, x, y, z)
    
    # 聚合体素内的点（取均值）
    unique_coors, inverse_indices = np.unique(coors_with_batch, axis=0, return_inverse=True)
    num_voxels = len(unique_coors)
    
    # 计算每个体素的特征均值和点数量
    voxels = np.zeros((num_voxels, num_features), dtype=np.float32)
    num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)
    for i in range(num_voxels):
        mask_voxel = inverse_indices == i
        voxels[i] = np.mean(features[mask_voxel], axis=0)
        num_points_per_voxel[i] = np.sum(mask_voxel)
    
    return voxels, unique_coors, num_points_per_voxel


def test_factorized_weight_subm_conv3d():
    # 1. 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查CUDA是否可用并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 2. 创建测试数据：随机生成一些3D点云数据
    num_points = 100
    points = np.random.rand(num_points, 6)  # [x, y, z, feature1, feature2, feature3]
    points[:, :3] *= 10  # 坐标范围放大到0-10
    
    # 3. 将点云转换为稀疏卷积需要的格式（使用手动体素化）
    voxel_size = [1, 1, 1]
    coors_range = [0, 0, 0, 10, 10, 10]  # [xmin, ymin, zmin, xmax, ymax, zmax]
    voxels, coors, num_points_per_voxel = point_cloud_to_voxel(points, voxel_size, coors_range)
    
    # 转换为稀疏卷积张量并移动到设备
    coors = torch.from_numpy(coors).int().to(device)
    features = torch.from_numpy(voxels).float().to(device)
    spatial_shape = [10, 10, 10]  # 空间形状（与坐标范围和体素大小匹配）
    batch_size = 1  # 批次大小
    input_tensor = spconv.SparseConvTensor(
        features, coors, spatial_shape, batch_size
    )
    
    # 4. 创建两个卷积层实例进行对比（移动到设备）
    # 实例1: weight_factors参与训练
    conv_trainable = FactorizedWeightSubMConv3d(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        factors_requires_grad=True  # 参与训练
    ).to(device)
    
    # 实例2: weight_factors不参与训练
    conv_non_trainable = FactorizedWeightSubMConv3d(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        factors_requires_grad=False  # 不参与训练
    ).to(device)
    
    # 复制权重，使两个网络初始状态相同
    conv_non_trainable.weight.data = conv_trainable.weight.data.clone()
    conv_non_trainable.weight_factors[0].data = conv_trainable.weight_factors[0].data.clone()
    conv_non_trainable.weight_factors[1].data = conv_trainable.weight_factors[1].data.clone()
    conv_non_trainable.weight_factors[2].data = conv_trainable.weight_factors[2].data.clone()
    if conv_trainable.bias is not None and conv_non_trainable.bias is not None:
        conv_non_trainable.bias.data = conv_trainable.bias.data.clone()
    
    # 5. 保存初始的weight_factors用于后续对比
    initial_factors_trainable = [
        conv_trainable.weight_factors[0].data.clone(),
        conv_trainable.weight_factors[1].data.clone(),
        conv_trainable.weight_factors[2].data.clone()
    ]
    
    initial_factors_non_trainable = [
        conv_non_trainable.weight_factors[0].data.clone(),
        conv_non_trainable.weight_factors[1].data.clone(),
        conv_non_trainable.weight_factors[2].data.clone()
    ]
    
    # 6. 定义优化器和损失函数
    optimizer_trainable = torch.optim.SGD(conv_trainable.parameters(), lr=0.01)
    optimizer_non_trainable = torch.optim.SGD(conv_non_trainable.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # 7. 创建目标输出（随机生成独立目标，确保损失不为零）
    with torch.no_grad():
        # 获取输出特征形状
        dummy_output = conv_trainable(input_tensor).features
        target_trainable = torch.randn_like(dummy_output)  # 随机目标
        target_non_trainable = torch.randn_like(conv_non_trainable(input_tensor).features)
    
    # 8. 进行一次训练迭代
    # 训练第一个网络（weight_factors参与训练）
    optimizer_trainable.zero_grad()
    output_trainable = conv_trainable(input_tensor).features
    loss_trainable = criterion(output_trainable, target_trainable)
    print(f"训练损失值: {loss_trainable.item()}")  # 确认损失不为零
    loss_trainable.backward()
    optimizer_trainable.step()
    
    # 训练第二个网络（weight_factors不参与训练）
    optimizer_non_trainable.zero_grad()
    output_non_trainable = conv_non_trainable(input_tensor).features
    loss_non_trainable = criterion(output_non_trainable, target_non_trainable)
    loss_non_trainable.backward()
    optimizer_non_trainable.step()
    
    # 9. 检查结果
    print("测试结果:")
    
    # 检查可训练的weight_factors是否发生了变化
    factors_changed_trainable = [
        not torch.allclose(conv_trainable.weight_factors[i].data, initial_factors_trainable[i])
        for i in range(3)
    ]
    print(f"可训练因子是否发生变化: {factors_changed_trainable}")  # 预期: [True, True, True]
    
    # 检查不可训练的weight_factors是否保持不变
    factors_unchanged_non_trainable = [
        torch.allclose(conv_non_trainable.weight_factors[i].data, initial_factors_non_trainable[i])
        for i in range(3)
    ]
    print(f"不可训练因子是否保持不变: {factors_unchanged_non_trainable}")  # 预期: [True, True, True]
    
    # 检查卷积核权重是否都发生了变化
    weight_changed_trainable = not torch.allclose(conv_trainable.weight.data, conv_non_trainable.weight.data)
    print(f"卷积核权重是否发生变化: {weight_changed_trainable}")  # 预期: True
    
    return factors_changed_trainable, factors_unchanged_non_trainable, weight_changed_trainable

import torch



if __name__ == "__main__":
    # 运行测试
    test_factorized_weight_subm_conv3d()
