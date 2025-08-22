import unittest
import torch
import spconv.pytorch as spconv
import numpy as np

# 假设被测试的类在factorized_conv.py中
from network.cnn_mod import FactorizedKernelSubMConv3d  # 请替换为实际模块名

class TestFactorizedKernelSubMConv3d(unittest.TestCase):
    def setUp(self):
        """准备测试数据和通用参数"""
        self.in_channels = 8
        self.out_channels = 16
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.groups = 2
        
        # 创建测试用的稀疏张量
        # 稀疏张量参数: 3D空间, 10个活跃点
        self.features = torch.randn(10, self.in_channels)  # 10个点, 每个点8个特征
        self.indices = torch.randint(0, 16, (10, 4), dtype=torch.int32)  # (batch_idx, z, y, x)
        self.indices[:, 0] = 0  # 所有点都属于第0个batch
        self.spatial_shape = (16, 16, 16)  # 3D空间大小
        self.batch_size = 1
        
        self.sparse_input = spconv.SparseConvTensor(
            features=self.features,
            indices=self.indices,
            spatial_shape=self.spatial_shape,
            batch_size=self.batch_size
        )
    
    def test_initialization(self):
        """测试初始化功能"""
        # 测试int类型kernel_size
        conv = FactorizedKernelSubMConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )
        
        # 验证kernel_size正确转换为列表
        self.assertEqual(conv.kernel_size, [3, 3, 3])
        
        # 验证参数存在
        self.assertTrue(hasattr(conv, 'factor_d'))
        self.assertTrue(hasattr(conv, 'factor_h'))
        self.assertTrue(hasattr(conv, 'factor_w'))
        
        # 验证父类weight已被删除
        self.assertFalse(hasattr(conv, 'weight'))
        
        # 测试list类型kernel_size
        conv = FactorizedKernelSubMConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=[2, 3, 4],
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )
        self.assertEqual(conv.kernel_size, [2, 3, 4])
        self.assertEqual(conv.factor_d.shape, (2,))
        self.assertEqual(conv.factor_h.shape, (3,))
        self.assertEqual(conv.factor_w.shape, (4,))
    
    def test_weight_generation(self):
        """测试权重生成功能"""
        conv = FactorizedKernelSubMConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=[2, 2, 2],
            groups=self.groups
        )
        
        # 手动设置因子值以便验证
        conv.factor_d.data = torch.tensor([1.0, 2.0])
        conv.factor_h.data = torch.tensor([3.0, 4.0])
        conv.factor_w.data = torch.tensor([5.0, 6.0])
        
        # 生成完整权重
        full_weight = conv._generate_full_weight()
        
        # 验证权重形状
        expected_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            2, 2, 2
        )
        self.assertEqual(full_weight.shape, expected_shape)
        
        # 验证权重值 (应该是三个因子的乘积)
        # 检查第一个输出通道和第一个输入通道组的权重
        expected_kernel = torch.tensor([
            [
                [1.0*3.0*5.0, 1.0*3.0*6.0],
                [1.0*4.0*5.0, 1.0*4.0*6.0]
            ],
            [
                [2.0*3.0*5.0, 2.0*3.0*6.0],
                [2.0*4.0*5.0, 2.0*4.0*6.0]
            ]
        ])
        
        # 所有输出通道和输入通道组的权重应该相同(因为是重复的)
        for out_c in range(self.out_channels):
            for in_g in range(self.in_channels // self.groups):
                torch.testing.assert_close(
                    full_weight[out_c, in_g], 
                    expected_kernel,
                    rtol=1e-6,
                    atol=1e-6
                )
    
    def test_forward_pass(self):
        """测试前向传播"""
        conv = FactorizedKernelSubMConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )
        
        # 执行前向传播
        output = conv(self.sparse_input)
        
        # 验证输出类型和形状
        self.assertIsInstance(output, spconv.SparseConvTensor)
        self.assertEqual(output.features.shape[1], self.out_channels)  # 特征数应等于out_channels
        self.assertEqual(output.batch_size, self.batch_size)
        self.assertEqual(len(output.spatial_shape), 3)  # 保持3D空间
    
    def test_parameter_count(self):
        """测试参数数量是否正确"""
        kernel_size = [2, 3, 4]
        conv = FactorizedKernelSubMConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            groups=self.groups
        )
        
        # 计算总参数数量
        total_params = sum(p.numel() for p in conv.parameters())
        
        # 预期参数数量: 三个因子的长度之和 (如果有偏置则加上偏置数量)
        expected_params = sum(kernel_size)
        if conv.bias is not None:
            expected_params += self.out_channels
            
        self.assertEqual(total_params, expected_params)
    
    def test_custom_weight_factors(self):
        """测试使用自定义权重因子初始化"""
        # 创建自定义权重因子
        factor_d = torch.tensor([1.0, 2.0])
        factor_h = torch.tensor([3.0, 4.0])
        factor_w = torch.tensor([5.0, 6.0])
        
        # 使用自定义因子初始化
        conv = FactorizedKernelSubMConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=[2, 2, 2],
            weight_factors=(factor_d, factor_h, factor_w),
            groups=self.groups
        )
        
        # 验证因子是否正确设置
        torch.testing.assert_close(conv.factor_d, factor_d)
        torch.testing.assert_close(conv.factor_h, factor_h)
        torch.testing.assert_close(conv.factor_w, factor_w)
    
    def test_exception_handling(self):
        """测试异常处理"""
        # 测试kernel_size维度错误
        with self.assertRaises(AssertionError):
            FactorizedKernelSubMConv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=[3, 3],  # 二维而不是三维
                groups=self.groups
            )
        
        # 测试权重因子形状不匹配
        with self.assertRaises(AssertionError):
            FactorizedKernelSubMConv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=[2, 2, 2],
                weight_factors=(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0])),  # 形状不匹配
                groups=self.groups
            )
        
        # 测试groups不整除in_channels
        with self.assertRaises(AssertionError):
            FactorizedKernelSubMConv3d(
                in_channels=7,  # 7不能被2整除
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                groups=2
            )

if __name__ == '__main__':
    unittest.main()