import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List

# class SubMConv3d(nn.Conv3d):
#     """简化的子流形卷积类，继承自PyTorch的Conv3d"""
#     pass  # 直接使用父类的forward，不做修改

class FactorizedWeightConv3d(nn.Conv3d):
    """基于因子分解的权重矩阵子流形卷积层"""
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
            assert len(weight_factors) == 3, "weight_factors must be a tuple of 3 tensors"
            assert weight_factors[0].shape == (k_d,), f"First factor must have length {k_d}"
            assert weight_factors[1].shape == (k_h,), f"Second factor must have length {k_h}"
            assert weight_factors[2].shape == (k_w,), f"Third factor must have length {k_w}"
            # return [torch.nn.Parameter(factor.clone()) for factor in weight_factors]
            return nn.ParameterList([torch.nn.Parameter(factor.clone()) for factor in weight_factors])
        else:
            # 初始化为随机值（而非全1，更容易观察变化）
            # return nn.ParameterList([
            #     torch.nn.Parameter(torch.randn(k_d)),
            #     torch.nn.Parameter(torch.randn(k_h)),
            #     torch.nn.Parameter(torch.randn(k_w))
            # ]
            return nn.ParameterList([
            torch.nn.Parameter(torch.randn(k_d)),
            torch.nn.Parameter(torch.randn(k_h)),
            torch.nn.Parameter(torch.randn(k_w))
            ])
    
    def _generate_weight_matrix(self):
        """生成三维权重矩阵 M[i,j,k] = a[0][i] * a[1][j] * a[2][k]"""
        a_d, a_h, a_w = self.weight_factors
        # print(f"a_d: ", a_d)
        # print(f"a_h: ", a_h)
        # print(f"a_w: ", a_w)
        return a_d.view(-1, 1, 1) * a_h.view(1, -1, 1) * a_w.view(1, 1, -1)
    
    def forward(self, input):
        # 1. 生成因子权重矩阵
        weight_matrix = self._generate_weight_matrix()  # 形状: (k_d, k_h, k_w)
        
        # 2. 扩展因子矩阵形状，与原始卷积核相乘
        # 原始卷积核形状: [out_channels, in_channels//groups, k_d, k_h, k_w]
        weight_matrix_expanded = weight_matrix.view(1, 1, *self.kernel_size)  # 扩展为 [1, 1, k_d, k_h, k_w]
        weighted_weight = self.weight * weight_matrix_expanded  # 逐元素相乘，保持卷积核形状
        
        # 3. 手动调用F.conv3d执行卷积（不修改原始权重，保持计算图完整）
        output = F.conv3d(
            input,
            weight=weighted_weight,  # 使用生成的权重
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        return output  # 直接返回结果，不修改原始权重
    
    def get_weight_factors(self):
        """获取三个维度的因子数组"""
        # a_d, a_h, a_w = self.weight_factors
        # print(f"a_d: ", a_d)
        # print(f"a_h: ", a_h)
        # print(f"a_w: ", a_w)
        return self.weight_factors


def test_factorized_weight_subm_conv3d():
    torch.manual_seed(42)  # 固定随机种子
    
    # 1. 创建测试数据 (batch=2, channels=3, depth=10, height=10, width=10)
    input_tensor = torch.randn(2, 3, 10, 10, 10)

    factor_d = torch.tensor([1.0, 1.2, 1.5])  # 假设深度维度为3
    factor_h = torch.tensor([1.0, 1.2, 1.5])  # 假设高度维度为3
    factor_w = torch.tensor([1.0, 1.2, 1.5])  # 假设宽度维度为3

    
    # 2. 创建两个卷积层实例
    conv_trainable = FactorizedWeightConv3d(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        weight_factors=(factor_d, factor_h, factor_w),
        factors_requires_grad=True  # 可训练
    )
    
    
    print(" ***************************  ")
    
    conv_non_trainable = FactorizedWeightConv3d(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        factors_requires_grad=False  # 不可训练
    )
    
    # 复制初始权重，保证对比公平性
    conv_non_trainable.load_state_dict(conv_trainable.state_dict())
    
    # 3. 保存初始因子和权重用于对比
    initial_factors_trainable = [
        conv_trainable.weight_factors[i].data.clone() for i in range(3)
    ]
    initial_factors_non_trainable = [
        conv_non_trainable.weight_factors[i].data.clone() for i in range(3)
    ]
    initial_weight_trainable = conv_trainable.weight.data.clone()
    initial_weight_non_trainable = conv_non_trainable.weight.data.clone()
    
    # 4. 定义优化器和损失函数
    optimizer_trainable = torch.optim.SGD(conv_trainable.parameters(), lr=0.1)
    optimizer_non_trainable = torch.optim.SGD(conv_non_trainable.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    
    # 5. 生成随机目标张量
    with torch.no_grad():
        target_shape = conv_trainable(input_tensor).shape
        target_trainable = torch.randn(*target_shape)
        target_non_trainable = torch.randn(*target_shape)
    
    # 6. 进行多次训练迭代
    for _ in range(10):
        # 训练可训练因子的网络
        optimizer_trainable.zero_grad()
        output_trainable = conv_trainable(input_tensor)
        loss_trainable = criterion(output_trainable, target_trainable)
        loss_trainable.backward()
        optimizer_trainable.step()
        
        # 训练不可训练因子的网络
        optimizer_non_trainable.zero_grad()
        output_non_trainable = conv_non_trainable(input_tensor)
        loss_non_trainable = criterion(output_non_trainable, target_non_trainable)
        loss_non_trainable.backward()
        optimizer_non_trainable.step()
    
    # 7. 验证结果
    print("测试结果:")
    
    # 可训练因子是否变化
    factors_changed_trainable = [
        not torch.allclose(conv_trainable.weight_factors[i].data, initial_factors_trainable[i], atol=1e-5)
        for i in range(3)
    ]
    factors_trainable = [
        conv_trainable.weight_factors[i].data.clone() for i in range(3)
    ]
    print(f"initial_factors_trainable: {initial_factors_trainable}")
    print(f"factors_trainable: {factors_trainable}")
    
    print(f"可训练因子是否发生变化: {factors_changed_trainable}")  # 预期: [True, True, True]
    
    # 不可训练因子是否不变
    factors_unchanged_non_trainable = [
        torch.allclose(conv_non_trainable.weight_factors[i].data, initial_factors_non_trainable[i], atol=1e-5)
        for i in range(3)
    ]
    print(f"不可训练因子是否保持不变: {factors_unchanged_non_trainable}")  # 预期: [True, True, True]
    
    # 卷积核权重是否变化
    weight_changed_trainable = not torch.allclose(
        conv_trainable.weight.data, initial_weight_trainable, atol=1e-5
    )
    weight_changed_non_trainable = not torch.allclose(
        conv_non_trainable.weight.data, initial_weight_non_trainable, atol=1e-5
    )
    print(f"可训练网络的卷积核是否变化: {weight_changed_trainable}")  # 预期: True
    print(f"不可训练网络的卷积核是否变化: {weight_changed_non_trainable}")  # 预期: True


if __name__ == "__main__":
    test_factorized_weight_subm_conv3d()
