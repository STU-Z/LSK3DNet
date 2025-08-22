import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F
# from spconv import SubMConv3d
from spconv.pytorch import SubMConv3d
from typing import Optional, Union, List, Tuple

class LearnableWeightSubMConv3d(SubMConv3d):
    """
    权重矩阵为可学习参数的子流形卷积层
    
    卷积核会与一个可学习的权重矩阵逐元素相乘，权重矩阵的形状与卷积核的空间尺寸一致
    
    权重矩阵参数参与训练
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
                 weight_init: str = "ones",  # 权重矩阵初始化方式
                 weight_regularizer: Optional[float] = None,  # L2正则化系数
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,** kwargs
        )
        
        # 处理kernel_size为列表形式
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 3
        else:
            assert len(kernel_size) == 3, "kernel_size must be 3D for 3D convolution"
            self.kernel_size = kernel_size
        
        # 初始化可学习的权重矩阵
        self.weight_matrix = self._initialize_weight_matrix(weight_init)
        self.weight_matrix = torch.nn.Parameter(self.weight_matrix, requires_grad=True)
        
        # 存储正则化系数（用于自定义损失计算）
        self.weight_regularizer = weight_regularizer
    
    def _initialize_weight_matrix(self, init_method: str) -> torch.Tensor:
        """初始化权重矩阵的不同方式"""
        k_d, k_h, k_w = self.kernel_size
        
        if init_method == "ones":
            # 初始化为全1矩阵（等价于原始卷积）
            return torch.ones(k_d, k_h, k_w)
        elif init_method == "uniform":
            # 均匀分布初始化
            return torch.nn.init.uniform_(torch.empty(k_d, k_h, k_w), 0.8, 1.2)
        elif init_method == "gaussian":
            # 高斯分布初始化（均值1.0，标准差0.1）
            return torch.nn.init.normal_(torch.empty(k_d, k_h, k_w), mean=1.0, std=0.1)
        elif init_method == "center-focused":
            # 中心聚焦初始化（中心权重为1，边缘衰减）
            weight = torch.ones(k_d, k_h, k_w)
            center = (k_d//2, k_h//2, k_w//2)
            for d in range(k_d):
                for h in range(k_h):
                    for w in range(k_w):
                        # 距离中心越远，权重越小
                        dist = abs(d - center[0]) + abs(h - center[1]) + abs(w - center[2])
                        weight[d, h, w] = 1.0 / (1.0 + 0.1 * dist)
            return weight
        else:
            raise ValueError(f"不支持的初始化方式: {init_method}")
    
    def forward(self, input: spconv.SparseConvTensor):
        # 获取原始卷积核权重 [out_channels, in_channels//groups, k_d, k_h, k_w]
        original_weight = self.weight
        
        # 扩展权重矩阵形状以匹配卷积核 [1, 1, k_d, k_h, k_w]
        weight_matrix_expanded = self.weight_matrix.view(1, 1, *self.kernel_size)
        
        # 逐元素相乘得到新的卷积核
        weighted_weight = original_weight * weight_matrix_expanded
        
        # 临时替换卷积核权重
        original_weight_data = self.weight.data
        self.weight.data = weighted_weight.data
        
        # 执行卷积计算
        output = super().forward(input)
        
        # 恢复原始权重
        self.weight.data = original_weight_data
        
        return output
    
    def get_regularization_loss(self) -> torch.Tensor:
        """计算权重矩阵的L2正则化损失（可选）"""
        if self.weight_regularizer is not None:
            return self.weight_regularizer * torch.norm(self.weight_matrix, p=2)
        return torch.tensor(0.0, device=self.weight_matrix.device)
    
    
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


class FactorizedWeightSubMConv3d(SubMConv3d):
    """
    基于因子分解的权重矩阵子流形卷积层
    
    权重矩阵M[i,j,k]由三个一维数组a[0], a[1], a[2]生成，其中M[i,j,k] = a[0][i] * a[1][j] * a[2][k]
    
    权重矩阵参数参与训练: factors_requires_grad: bool = False
   
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
    
    def forward(self, input: spconv.SparseConvTensor):
        # 1. 生成因子权重矩阵
        weight_matrix = self._generate_weight_matrix()  # 形状: (k_d, k_h, k_w)
        
        # 2. 扩展因子矩阵形状，与原始卷积核相乘
        weight_matrix_expanded = weight_matrix.view(1, 1, *self.kernel_size)  # [1, 1, k_d, k_h, k_w]
        weighted_weight = self.weight * weight_matrix_expanded  # 逐元素相乘
        
        # 3. 保存原始权重，使用新权重进行计算，之后恢复原始权重
        original_weight = self.weight.data
        self.weight.data = weighted_weight.data  # 临时替换权重
        
        # 调用父类的forward方法执行卷积
        output = super().forward(input)
        
        # 恢复原始权重
        self.weight.data = original_weight
        
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

# # ######################## 使用示例 ######################## #

# # 创建三个因子数组
# factor_d = torch.tensor([1.0, 1.2, 1.5, 1.2, 1.0])  # 假设深度维度为5
# factor_h = torch.tensor([1.0, 1.2, 1.5, 1.2, 1.0])  # 假设高度维度为5
# factor_w = torch.tensor([1.0, 1.2, 1.5, 1.2, 1.0])  # 假设宽度维度为5

# # 创建卷积层
# conv = FactorizedWeightSubMConv3d(
#     in_channels=32,
#     out_channels=64,
#     kernel_size=5,
#     weight_factors=(factor_d, factor_h, factor_w),
#     factors_requires_grad=False  # 关键参数：设置为False使weight_factors不参与训练
# )



### ********************************************************************* ###

# 这个实现与之前版本的核心区别在于：卷积核的所有空间参数完全由三个一维数组生成，
# 没有原始的self.weight参数，因此整个卷积层的可学习参数只有 3k 个（k 为卷积核的空间尺寸）。


#你的方法的独特性
#在 3D 稀疏卷积（如spconv场景）中，你的实现有其特殊性：

#针对稀疏数据设计，保持了SubMConv3d的子流形特性（不引入新的非零元素）。
#分解方式更直接（乘积形式），避免了可分离卷积的多步串联计算，更适合稀疏场景的高效实现。
#参数量减少更显著（尤其当 k 较大时），且保留了各向异性的学习能力（三个维度的因子独立学习）。
#总结
#类似的 “卷积核因子分解” 思想在深度学习中已有广泛研究，尤其是在模型压缩、高效推理领域。你的方法是这些思想在 3D 稀疏卷积场景下的具体实现，核心思路与低秩分解、
#可分离卷积一脉相承，但针对稀疏数据的特性做了适配，具有明确的理论基础和实践价值。


class FactorizedKernelSubMConv3d(SubMConv3d):
    """
    卷积核完全由三个一维数组构成的子流形卷积层
    
    卷积核的空间参数通过 M[i,j,k] = a[0][i] × a[1][j] × a[2][k] 生成，
    整个卷积层的可学习参数仅为这三个一维数组（共3k个参数）
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
                 weight_factors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,** kwargs):
        # 先调用父类构造函数，后续会替换weight参数
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
        
        # 验证分组参数的兼容性
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        self.groups = groups
        self.in_channels_per_group = in_channels // groups
        
        # 初始化三个维度的因子数组（这些是唯一的可学习参数）
        self.weight_factors = self._initialize_weight_factors(weight_factors, k_d, k_h, k_w)
        
        # 移除父类的weight参数，改用我们的因子数组
        if hasattr(self, 'weight'):
            del self.weight
        
        # 注册因子数组为模块参数
        self.register_parameter('factor_d', self.weight_factors[0])
        self.register_parameter('factor_h', self.weight_factors[1])
        self.register_parameter('factor_w', self.weight_factors[2])
    
    def _initialize_weight_factors(self, weight_factors, k_d, k_h, k_w):
        """初始化三个维度的因子数组（这些是唯一的可学习参数）"""
        if weight_factors is not None:
            # 检查输入的因子是否符合要求
            assert len(weight_factors) == 3, "weight_factors must be a tuple of 3 tensors"
            assert weight_factors[0].shape == (k_d,), f"First factor must have length {k_d}"
            assert weight_factors[1].shape == (k_h,), f"Second factor must have length {k_h}"
            assert weight_factors[2].shape == (k_w,), f"Third factor must have length {k_w}"
            
            return [torch.nn.Parameter(factor.clone()) for factor in weight_factors]
        else:
            # 使用Kaiming初始化因子数组
            factor_d = torch.empty(k_d)
            factor_h = torch.empty(k_h)
            factor_w = torch.empty(k_w)
            
            torch.nn.init.kaiming_uniform_(factor_d, a=torch.sqrt(5))
            torch.nn.init.kaiming_uniform_(factor_h, a=torch.sqrt(5))
            torch.nn.init.kaiming_uniform_(factor_w, a=torch.sqrt(5))
            
            return [
                torch.nn.Parameter(factor_d),
                torch.nn.Parameter(factor_h),
                torch.nn.Parameter(factor_w)
            ]
    
    def _generate_full_weight(self):
        """生成完整的卷积核权重矩阵"""
        # 生成空间权重矩阵 M[i,j,k] = a[0][i] × a[1][j] × a[2][k]
        spatial_weight = self.factor_d.view(-1, 1, 1) * self.factor_h.view(1, -1, 1) * self.factor_w.view(1, 1, -1)
        
        # 扩展为完整卷积核形状: [out_channels, in_channels//groups, k_d, k_h, k_w]
        # 这里通过重复操作将空间权重应用到所有输入输出通道
        full_weight = spatial_weight.view(1, 1, *self.kernel_size)
        full_weight = full_weight.repeat(self.out_channels, self.in_channels_per_group, 1, 1, 1)
        
        return full_weight
    
    def forward(self, input: spconv.SparseConvTensor):
        # 生成完整的卷积核权重
        full_weight = self._generate_full_weight()
        
        # 保存原始权重（这里实际上父类的weight已被删除，主要是为了兼容父类forward的预期）
        # 临时设置权重
        self.weight = full_weight
        
        # 执行卷积计算
        output = super().forward(input)
        
        # 清除临时权重
        del self.weight
        
        return output
    
    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, '
                f'groups={self.groups}, bias={self.bias is not None}')
    