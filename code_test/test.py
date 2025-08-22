import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
def fill_tensor(tensor, n, m):
    """
    处理一维torch.Tensor数组，对中心区域和其他区域进行不同方式的填充
    
    参数:
        tensor: 输入的一维torch.Tensor，形状为(k,)
        n: 中心比例，1>=n>=0
        m: 中心区域的权重值  1>=m>=0
        
    返回:
        填充后的torch.Tensor，形状与输入相同
    """
    # 获取张量长度
    k = tensor.size(0)
    
    # 计算中心区域的大小d（向下取整）
    d = int(k * n)
    
    # 确保d在有效范围内
    d = max(0, min(d, k))
    
    # 计算填充值
    if d > 0:
        center_value = m / d
    else:
        center_value = 0.0  # 当d为0时，中心区域无元素
    
    # 计算其他区域的总权重和单个填充值
    remaining_weight = 1.0 - m
    remaining_count = k - d
    
    if remaining_count > 0:
        other_value = remaining_weight / remaining_count
    else:
        other_value = 0.0  # 当没有其他区域时
    
    # 创建填充结果张量
    result = torch.empty_like(tensor)
    
    # 计算中心区域的起始和结束索引
    start_idx = (k - d) // 2
    end_idx = start_idx + d
    
    # 填充中心区域
    result[start_idx:end_idx] = center_value
    
    # 填充其他区域（前半部分）
    if start_idx > 0:
        result[:start_idx] = other_value
    
    # 填充其他区域（后半部分）
    if end_idx < k:
        result[end_idx:] = other_value
    # 归一化处理：确保总和为1（处理可能的浮点误差）
    total = result.sum()
    if total != 0:  # 避免除以零
        result = result / total
    return result



def fill_3d_matrix(shape, n, m):
    """
    对三维矩阵进行填充，中心区域分配权重m，其他区域分配权重1-m
    
    参数:
        shape: 三维矩阵的形状，元组形式 (k_d, k_h, k_w)
        n: 中心区域在每个维度上的比例，1>=n>=0
        m: 中心区域的总权重，1>=m>=0
        
    返回:
        填充后的三维torch.Tensor，形状为输入的shape
    """
    k_d, k_h, k_w = shape
    
    # 计算每个维度上中心区域的大小（向下取整）
    d_d = int(k_d * n)
    d_h = int(k_h * n)
    d_w = int(k_w * n)
    
    # 确保每个维度的中心区域大小在有效范围内
    d_d = max(0, min(d_d, k_d))
    d_h = max(0, min(d_h, k_h))
    d_w = max(0, min(d_w, k_w))
    
    # 计算中心区域的总元素数量
    center_elements = d_d * d_h * d_w
    
    # 计算中心区域和其他区域的填充值
    if center_elements > 0:
        center_value = m / center_elements
    else:
        center_value = 0.0  # 当中心区域无元素时
    
    # 计算其他区域的总元素数量和填充值
    total_elements = k_d * k_h * k_w
    other_elements = total_elements - center_elements
    
    if other_elements > 0:
        other_value = (1.0 - m) / other_elements
    else:
        other_value = 0.0  # 当没有其他区域时
    
    # 创建填充结果张量
    result = torch.empty(shape)
    
    # 计算每个维度上中心区域的起始和结束索引
    start_d = (k_d - d_d) // 2
    end_d = start_d + d_d
    
    start_h = (k_h - d_h) // 2
    end_h = start_h + d_h
    
    start_w = (k_w - d_w) // 2
    end_w = start_w + d_w
    
    # 填充中心区域
    result[start_d:end_d, start_h:end_h, start_w:end_w] = center_value
    
    # 填充其他区域（使用掩码操作）
    # 创建中心区域掩码
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[start_d:end_d, start_h:end_h, start_w:end_w] = True
    
    # 对非中心区域填充
    result[~mask] = other_value
    
    # 归一化处理：确保总和为1（处理可能的浮点误差）
    total = result.sum()
    if total != 0:
        result = result / total
    
    return result


if __name__ == "__main__":
    k_d, k_h, k_w = 9, 9, 9
    k=k_d
    factor_d = torch.randn(k_d)
    
    # 生成两个一维张量
    center_data = fill_tensor(factor_d, 0.6, 0.8)  # 填充中心区域
    edge_data = fill_tensor(factor_d, 0.6, 0.8)
    height_data= fill_tensor(factor_d, 0.6, 0.8)  # 填充高度区域
    print("中心填充:", center_data)
    print("边缘填充:", edge_data)
    
    
    # 方法1：使用外积计算 data_2d[i][j] = center_data[i] * edge_data[j]
    # 通过将center_data变为列向量，与edge_data行向量相乘
    # data_2d = torch.outer(center_data, edge_data)
    
    # 方法2：使用广播机制（等价于方法1）
    data_2d = center_data.unsqueeze(1) * edge_data.unsqueeze(0)
    
    data_3d = torch.einsum('i,j,k->ijk', center_data, edge_data, height_data)
    

    
    print(f"center_data形状: {center_data.shape}")
    print(f"edge_data形状: {edge_data.shape}")
    print(f"生成的9x9矩阵形状: {data_2d.shape}")
    print("\n9x9矩阵内容:")
    print(data_2d)
    
    # 验证其中一个元素
    i, j = 2, 3  # 任意索引
    print(f"\n验证: data_2d[{i}][{j}] = {data_2d[i][j]}, 计算值 = {center_data[i] * edge_data[j]}")
    
    # 归一化
    data_sum = data_2d.sum()
    if data_sum != 0:
        data_normalized_2d = data_2d / data_sum
    else:
        data_normalized_2d = data_2d
    
    # 可视化设置
    plt.figure(figsize=(10, 8))
    
    # 创建热力图，使用viridis配色方案（数值越大颜色越亮）
    # annot=True显示数值，fmt=".4f"设置数值格式，cmap设置颜色映射
    sns.heatmap(data_normalized_2d.numpy(), annot=True, fmt=".4f", cmap="viridis", 
                cbar=True, square=True, linewidths=.5)
    
    plt.title("9x9 data_2d Matrix Visualization (Brighter = Larger Value)", fontsize=14)
    plt.xlabel("Edge data_2d Index", fontsize=12)
    plt.ylabel("Center data_2d Index", fontsize=12)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()


    # # 对3D数据进行归一化
    # total = data_3d.sum()
    # if total != 0:
    #     data_3d_normalized = data_3d / total
    #     print(f"\n归一化后总和: {data_3d_normalized.sum():.6f}")
    # else:
    #     data_3d_normalized = data_3d

    # # 简单可视化：显示中间切片
    # plt.figure(figsize=(10, 8))
    # mid_slice = data_3d_normalized[:, :, k_d//2].numpy()  # 取中间层
    # plt.imshow(mid_slice, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Normalized Value')
    # plt.title(f'3D Tensor Middle Slice (z={k_d//2})')
    # plt.xlabel('j Index')
    # plt.ylabel('i Index')
    # plt.tight_layout()
    # plt.show()
    
    # 归一化
    total = data_3d.sum()
    if total != 0:
        data_3d_normalized = data_3d / total
    else:
        data_3d_normalized = data_3d
    
    data_3d_normalized=fill_3d_matrix((k, k, k), 0.6, 0.1)
    # 准备3D可视化数据
    # 生成网格坐标 (i, j, k)
    i, j, k_indices = np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing='ij')
    
    # 将坐标和数值转换为一维数组（适合散点图）
    x = i.flatten()
    y = j.flatten()
    z = k_indices.flatten()
    values = data_3d_normalized.numpy().flatten()  # 数值用于颜色映射
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D散点图，点的大小和颜色由数值决定
    # s=50表示点的大小，c=values表示颜色映射，cmap选择viridis（数值越大越亮）
    scatter = ax.scatter(x, y, z, s=50, c=values, cmap='viridis', alpha=0.8)
    
    # 添加颜色条，显示数值与颜色的对应关系
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Normalized Value', rotation=270, labelpad=20)
    
    # 设置标题和轴标签
    ax.set_title('3D Visualization of data_3d_normalized', fontsize=14)
    ax.set_xlabel('i Index', fontsize=12)
    ax.set_ylabel('j Index', fontsize=12)
    ax.set_zlabel('k Index', fontsize=12)
    
    # 设置坐标轴范围
    ax.set_xlim(0, k-1)
    ax.set_ylim(0, k-1)
    ax.set_zlim(0, k-1)
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()