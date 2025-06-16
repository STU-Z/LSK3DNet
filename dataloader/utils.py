'''
Author: Zhangrunbang 254616730@qq.com
Date: 2025-05-23 19:52:16
LastEditors: Zhangrunbang 254616730@qq.com
LastEditTime: 2025-05-23 19:55:18
FilePath: /LSK3DNet/dataloader/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np

instance_classes_kitti = [0, 1, 2, 3, 4, 5, 6, 7]

def swap(pt1, pt2, start_angle, end_angle, label1, label2):
    # 在两个点云样本之间，按照指定的水平角度区间，交换扇区内的点和标签，实现点云的混合增强，提升模型的泛化能力。这是 Polarmix 等点云增强方法的基础操作之一。
    # calculate horizontal angle for each point
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

    # select points in sector
    idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
    idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))

    # swap
    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out = np.concatenate((pt1_out, pt2[idx2]))
    pt2_out = np.delete(pt2, idx2, axis=0)
    pt2_out = np.concatenate((pt2_out, pt1[idx1]))
    
    label1_out = np.delete(label1, idx1, axis=0)
    label1_out = np.concatenate((label1_out, label2[idx2]))
    label2_out = np.delete(label2, idx2, axis=0)
    label2_out = np.concatenate((label2_out, label1[idx1]))
    assert pt1_out.shape[0] == label1_out.shape[0]
    assert pt2_out.shape[0] == label2_out.shape[0]

    return pt1_out, pt2_out, label1_out, label2_out
    # 使用 PolarMix（或类似的扇区混合增强）确实有可能导致如下情况：
    # 一辆车（或其他物体）的点云被扇区边界截断，一部分被替换成了另一帧的点云，导致单个物体出现“缺失”或“拼接”的现象。
    # 这样增强后，点云中可能会出现一些不符合真实物理结构的“奇怪形状”。
    # 为什么还要用这种增强？
    # 提升模型鲁棒性
    # 虽然会产生“非真实”的样本，但这能让模型学会在点云缺失、遮挡、拼接等极端情况下也能做出合理判断，提升泛化能力。
    # 数据多样性
    # 增强后的数据分布更广，能缓解过拟合，尤其在数据量有限时效果明显。
    # 实际场景中也常有遮挡/缺失
    # 真实自动驾驶场景下，点云经常被遮挡或部分丢失，模型需要适应这种情况。
    # 如何缓解“奇怪形状”带来的负面影响？
    # PolarMix 通常只对“thing类”（可实例分割的物体，如车、人等）做增强，减少对背景的影响。
    # 可以通过调整扇区角度、增强概率、只对部分类别做增强等方式，降低对结构的破坏。
    # 训练时模型会自动学习到哪些特征是“异常”的，推理时一般不会输出这些异常结构。
    # 总结：
    # PolarMix 等增强方法确实会带来点云截断和拼接，但这是有意为之，目的是提升模型鲁棒性和泛化能力。只要增强比例合适，通常不会对最终性能造成负面影响，反而能提升模型在复杂场景下的表现。

def rotate_copy(pts, labels, instance_classes, Omega):
    # extract instance points
    pts_inst, labels_inst = [], []
    for s_class in instance_classes:
        pt_idx = np.where((labels == s_class))
        pts_inst.append(pts[pt_idx])
        labels_inst.append(labels[pt_idx])
    pts_inst = np.concatenate(pts_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)

    if len(pts_inst) == 0:
        return None, None
    
    # rotate-copy
    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    for omega_j in Omega:
        rot_mat = np.array([[np.cos(omega_j),
                             np.sin(omega_j), 0],
                            [-np.sin(omega_j),
                             np.cos(omega_j), 0], [0, 0, 1]])
        new_pt = np.zeros_like(pts_inst)
        new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
        new_pt[:, 3] = pts_inst[:, 3]
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    return pts_copy, labels_copy

def polarmix(pts1, labels1, pts2, labels2, alpha, beta, instance_classes, Omega):
    pts_out, labels_out = pts1, labels1
    # swapping
    if np.random.random() < 0.5:
        pts_out, _, labels_out, _ = swap(pts1, pts2, start_angle=alpha, end_angle=beta, label1=labels1, label2=labels2)

    # rotate-pasting
    if np.random.random() < 1.0:
        # rotate-copy
        pts_copy, labels_copy = rotate_copy(pts2, labels2, instance_classes, Omega)
        # paste
        if pts_copy is not None:
            pts_out = np.concatenate((pts_out, pts_copy), axis=0)
            labels_out = np.concatenate((labels_out, labels_copy), axis=0)

    return pts_out, labels_out