import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np
from spconv.pytorch import SubMConv3d
from typing import Optional, Union, List, Tuple


class FactorGenerator(nn.Module):
    """独立的因子生成网络，负责生成三个维度的weight_factors"""
    def __init__(self, 
                 kernel_size: Union[int, List[int]],
                 weight_factors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                 requires_grad: bool = True):
        super().__init__()
        
        # 处理kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 3
        else:
            assert len(kernel_size) == 3, "kernel_size must be 3D for 3D convolution"
            self.kernel_size = kernel_size
        
        k_d, k_h, k_w = self.kernel_size
        
        # 初始化因子参数
        if weight_factors is not None:
            assert len(weight_factors) == 3, "weight_factors must be a tuple of 3 tensors"
            assert weight_factors[0].shape == (k_d,), f"First factor must have length {k_d}"
            assert weight_factors[1].shape == (k_h,), f"Second factor must have length {k_h}"
            assert weight_factors[2].shape == (k_w,), f"Third factor must have length {k_w}"
            
            self.factor_d = nn.Parameter(weight_factors[0].clone(), requires_grad=requires_grad)
            self.factor_h = nn.Parameter(weight_factors[1].clone(), requires_grad=requires_grad)
            self.factor_w = nn.Parameter(weight_factors[2].clone(), requires_grad=requires_grad)
        else:
            # 随机初始化
            self.factor_d = nn.Parameter(torch.randn(k_d), requires_grad=requires_grad)
            self.factor_h = nn.Parameter(torch.randn(k_h), requires_grad=requires_grad)
            self.factor_w = nn.Parameter(torch.randn(k_w), requires_grad=requires_grad)
    
    def forward(self):
        """生成并返回三个维度的因子"""
        return self.factor_d, self.factor_h, self.factor_w
    
    def get_factors(self):
        return self.factor_d, self.factor_h, self.factor_w
    
    def set_factors(self, new_factors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        k_d, k_h, k_w = self.kernel_size
        assert len(new_factors) == 3, "new_factors must be a tuple of 3 tensors"
        assert new_factors[0].shape == (k_d,), f"First factor must have length {k_d}"
        assert new_factors[1].shape == (k_h,), f"Second factor must have length {k_h}"
        assert new_factors[2].shape == (k_w,), f"Third factor must have length {k_w}"
        
        with torch.no_grad():
            self.factor_d.copy_(new_factors[0])
            self.factor_h.copy_(new_factors[1])
            self.factor_w.copy_(new_factors[2])


class FactorizedWeightSubMConv3d(SubMConv3d):
    """包含独立因子生成网络的子流形卷积层"""
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
        
        # 初始化因子生成网络
        self.factor_generator = FactorGenerator(
            kernel_size=kernel_size,
            weight_factors=weight_factors,
            requires_grad=factors_requires_grad
        )
        
        # 记录kernel_size用于后续维度检查
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 3
        else:
            self.kernel_size = kernel_size
    
    def _generate_weight_matrix(self, factors):
        """根据因子生成权重矩阵 M[i,j,k] = a[0][i] * a[1][j] * a[2][k]"""
        a_d, a_h, a_w = factors
        return a_d.view(-1, 1, 1) * a_h.view(1, -1, 1) * a_w.view(1, 1, -1)
    
    def forward(self, input: spconv.SparseConvTensor):
        # 1. 从独立网络获取因子
        factors = self.factor_generator()  # (a_d, a_h, a_w)
        
        # 2. 生成权重矩阵并扩展维度
        weight_matrix = self._generate_weight_matrix(factors)  # (k_d, k_h, k_w)
        weight_matrix_expanded = weight_matrix.view(1, 1, *self.kernel_size)  # [1,1,k_d,k_h,k_w]
        
        # 3. 计算最终卷积权重（原始权重 × 因子矩阵）
        weighted_weight = self.weight * weight_matrix_expanded
        
        # 4. 临时替换卷积权重并执行卷积
        original_weight = self.weight
        self.weight = torch.nn.Parameter(weighted_weight)
        output = super().forward(input)
        
        # 5. 恢复原始权重
        self.weight = original_weight
        
        return output
    
    def get_weight_factors(self):
        return self.factor_generator.get_factors()
    
    def set_weight_factors(self, new_factors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        self.factor_generator.set_factors(new_factors)


# 以下为测试代码（复用之前的点云转换和测试逻辑）
def point_cloud_to_voxel(points, voxel_size, coors_range):
    vsize_x, vsize_y, vsize_z = voxel_size
    x_min, y_min, z_min, x_max, y_max, z_max = coors_range
    
    coordinates = points[:, :3]
    features = points[:, 3:]
    num_features = features.shape[1]
    
    mask = (coordinates[:, 0] >= x_min) & (coordinates[:, 0] < x_max) & \
           (coordinates[:, 1] >= y_min) & (coordinates[:, 1] < y_max) & \
           (coordinates[:, 2] >= z_min) & (coordinates[:, 2] < z_max)
    coordinates = coordinates[mask]
    features = features[mask]
    if len(coordinates) == 0:
        return np.zeros((0, num_features), dtype=np.float32), np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.int32)
    
    voxel_indices = np.floor((coordinates - [x_min, y_min, z_min]) / [vsize_x, vsize_y, vsize_z]).astype(np.int32)
    batch_indices = np.zeros(len(voxel_indices), dtype=np.int32)
    coors_with_batch = np.column_stack([batch_indices, voxel_indices])
    
    unique_coors, inverse_indices = np.unique(coors_with_batch, axis=0, return_inverse=True)
    num_voxels = len(unique_coors)
    
    voxels = np.zeros((num_voxels, num_features), dtype=np.float32)
    num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)
    for i in range(num_voxels):
        mask_voxel = inverse_indices == i
        voxels[i] = np.mean(features[mask_voxel], axis=0)
        num_points_per_voxel[i] = np.sum(mask_voxel)
    
    return voxels, unique_coors, num_points_per_voxel


def test_factorized_weight_subm_conv3d():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    num_points = 100
    points = np.random.rand(num_points, 6)
    points[:, :3] *= 10
    
    voxel_size = [1, 1, 1]
    coors_range = [0, 0, 0, 10, 10, 10]
    voxels, coors, num_points_per_voxel = point_cloud_to_voxel(points, voxel_size, coors_range)
    
    coors = torch.from_numpy(coors).int().to(device)
    features = torch.from_numpy(voxels).float().to(device)
    spatial_shape = [10, 10, 10]
    batch_size = 1
    input_tensor = spconv.SparseConvTensor(features, coors, spatial_shape, batch_size)
    
    # 创建两个卷积层实例
    conv_trainable = FactorizedWeightSubMConv3d(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        factors_requires_grad=True
    ).to(device)
    
    conv_non_trainable = FactorizedWeightSubMConv3d(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        factors_requires_grad=False
    ).to(device)
    
    # 复制初始参数
    conv_non_trainable.weight.data = conv_trainable.weight.data.clone()
    conv_non_trainable.factor_generator.set_factors(conv_trainable.get_weight_factors())
    if conv_trainable.bias is not None and conv_non_trainable.bias is not None:
        conv_non_trainable.bias.data = conv_trainable.bias.data.clone()
    
    # 保存初始因子
    initial_factors_trainable = [
        conv_trainable.get_weight_factors()[0].data.clone(),
        conv_trainable.get_weight_factors()[1].data.clone(),
        conv_trainable.get_weight_factors()[2].data.clone()
    ]
    
    initial_factors_non_trainable = [
        conv_non_trainable.get_weight_factors()[0].data.clone(),
        conv_non_trainable.get_weight_factors()[1].data.clone(),
        conv_non_trainable.get_weight_factors()[2].data.clone()
    ]
    
    # 优化器和损失函数
    optimizer_trainable = torch.optim.SGD(conv_trainable.parameters(), lr=0.01)
    optimizer_non_trainable = torch.optim.SGD(conv_non_trainable.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # 创建目标输出
    with torch.no_grad():
        dummy_output = conv_trainable(input_tensor).features
        target_trainable = torch.randn_like(dummy_output)
        target_non_trainable = torch.randn_like(conv_non_trainable(input_tensor).features)
    
    # 训练迭代
    optimizer_trainable.zero_grad()
    output_trainable = conv_trainable(input_tensor).features
    loss_trainable = criterion(output_trainable, target_trainable)
    print(f"训练损失值: {loss_trainable.item()}")
    loss_trainable.backward()
    optimizer_trainable.step()
    
    optimizer_non_trainable.zero_grad()
    output_non_trainable = conv_non_trainable(input_tensor).features
    loss_non_trainable = criterion(output_non_trainable, target_non_trainable)
    loss_non_trainable.backward()
    optimizer_non_trainable.step()
    
    # 检查结果
    print("测试结果:")
    
    factors_changed_trainable = [
        not torch.allclose(conv_trainable.get_weight_factors()[i].data, initial_factors_trainable[i])
        for i in range(3)
    ]
    print(f"可训练因子是否发生变化: {factors_changed_trainable}")  # 预期: [True, True, True]
    
    factors_unchanged_non_trainable = [
        torch.allclose(conv_non_trainable.get_weight_factors()[i].data, initial_factors_non_trainable[i])
        for i in range(3)
    ]
    print(f"不可训练因子是否保持不变: {factors_unchanged_non_trainable}")  # 预期: [True, True, True]
    
    weight_changed_trainable = not torch.allclose(conv_trainable.weight.data, conv_non_trainable.weight.data)
    print(f"卷积核权重是否发生变化: {weight_changed_trainable}")  # 预期: True
    
    return factors_changed_trainable, factors_unchanged_non_trainable, weight_changed_trainable


if __name__ == "__main__":
    test_factorized_weight_subm_conv3d()
