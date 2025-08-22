'''
Author: Zhangrunbang 254616730@qq.com
Date: 2025-06-26 17:30:47
LastEditors: Zhangrunbang 254616730@qq.com
LastEditTime: 2025-07-28 11:33:37
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

# def rotate_copy(pts, labels, instance_classes, Omega):
#     # extract instance points
#     pts_inst, labels_inst = [], []
#     for s_class in instance_classes:
#         pt_idx = np.where((labels == s_class))
#         pts_inst.append(pts[pt_idx])
#         labels_inst.append(labels[pt_idx])
#     pts_inst = np.concatenate(pts_inst, axis=0)
#     labels_inst = np.concatenate(labels_inst, axis=0)

#     if len(pts_inst) == 0:
#         return None, None
    
#     # rotate-copy
#     pts_copy = [pts_inst]
#     labels_copy = [labels_inst]
#     for omega_j in Omega:
#         rot_mat = np.array([[np.cos(omega_j),
#                              np.sin(omega_j), 0],
#                             [-np.sin(omega_j),
#                              np.cos(omega_j), 0], [0, 0, 1]])
#         new_pt = np.zeros_like(pts_inst)
#         new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
#         new_pt[:, 3] = pts_inst[:, 3]
#         pts_copy.append(new_pt)
#         labels_copy.append(labels_inst)
#     pts_copy = np.concatenate(pts_copy, axis=0)
#     labels_copy = np.concatenate(labels_copy, axis=0)
#     return pts_copy, labels_copy

def rotate_copy(pts, labels, instance_classes, Omega):
    # extract instance points
    pts_inst, labels_inst = [], []
    for s_class in instance_classes:
        pt_idx = np.where((labels == s_class))
        pts_inst.append(pts[pt_idx])
        labels_inst.append(labels[pt_idx])
    if len(pts_inst) == 0 or sum([arr.shape[0] for arr in pts_inst]) == 0:
        return None, None
    pts_inst = np.concatenate(pts_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)
    if pts_inst.ndim != 2 or pts_inst.shape[0] == 0:
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
        if pts_inst.shape[1] > 3:
            new_pt[:, 3:] = pts_inst[:, 3:]
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    return pts_copy, labels_copy

# def polarmix(pts1, labels1, pts2, labels2, alpha, beta, instance_classes, Omega):
#     pts_out, labels_out = pts1, labels1
#     # swapping
#     if np.random.random() < 0.5:
#         pts_out, _, labels_out, _ = swap(pts1, pts2, start_angle=alpha, end_angle=beta, label1=labels1, label2=labels2)

#     # rotate-pasting
#     if np.random.random() < 1.0:
#         # rotate-copy
#         pts_copy, labels_copy = rotate_copy(pts2, labels2, instance_classes, Omega)
#         # paste
#         if pts_copy is not None:
#             pts_out = np.concatenate((pts_out, pts_copy), axis=0)
#             labels_out = np.concatenate((labels_out, labels_copy), axis=0)

#     return pts_out, labels_out


def swap_with_instance(pt1, pt2, start_angle, end_angle, label1, label2, inst1, inst2, sig1, sig2):
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

    idx1 = np.where((yaw1 > start_angle) & (yaw1 < end_angle))
    idx2 = np.where((yaw2 > start_angle) & (yaw2 < end_angle))

    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out = np.concatenate((pt1_out, pt2[idx2]), axis=0)
    pt2_out = np.delete(pt2, idx2, axis=0)
    pt2_out = np.concatenate((pt2_out, pt1[idx1]), axis=0)

    label1_out = np.delete(label1, idx1, axis=0)
    label1_out = np.concatenate((label1_out, label2[idx2]), axis=0)
    label2_out = np.delete(label2, idx2, axis=0)
    label2_out = np.concatenate((label2_out, label1[idx1]), axis=0)

    inst1_out = np.delete(inst1, idx1, axis=0).reshape(-1)
    inst1_out = np.concatenate((inst1_out, inst2[idx2].reshape(-1)), axis=0)
    inst2_out = np.delete(inst2, idx2, axis=0).reshape(-1)
    inst2_out = np.concatenate((inst2_out, inst1[idx1].reshape(-1)), axis=0)

    sig1_out = np.delete(sig1, idx1, axis=0).reshape(-1)
    sig1_out = np.concatenate((sig1_out, sig2[idx2].reshape(-1)), axis=0)
    sig2_out = np.delete(sig2, idx2, axis=0).reshape(-1)
    sig2_out = np.concatenate((sig2_out, sig1[idx1].reshape(-1)), axis=0)

    assert pt1_out.shape[0] == label1_out.shape[0] == inst1_out.shape[0] == sig1_out.shape[0]
    assert pt2_out.shape[0] == label2_out.shape[0] == inst2_out.shape[0] == sig2_out.shape[0]
    return pt1_out, pt2_out, label1_out, label2_out, inst1_out, inst2_out, sig1_out, sig2_out

def rotate_copy_with_instance(pts, labels, inst, sig, instance_classes, Omega):
    pts_inst, labels_inst, inst_inst, sig_inst = [], [], [], []
    for s_class in instance_classes:
        pt_idx = np.where((labels == s_class))
        pts_inst.append(pts[pt_idx])
        labels_inst.append(labels[pt_idx])
        inst_inst.append(inst[pt_idx[0]])
        sig_inst.append(sig[pt_idx[0]])
    if len(pts_inst) == 0 or sum([arr.shape[0] for arr in pts_inst]) == 0:
        return None, None, None, None
    pts_inst = np.concatenate(pts_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)
    inst_inst = np.concatenate(inst_inst, axis=0)
    sig_inst = np.concatenate(sig_inst, axis=0)
    if pts_inst.ndim != 2 or pts_inst.shape[0] == 0:
        return None, None, None, None

    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    inst_copy = [inst_inst]
    sig_copy = [sig_inst]
    for omega_j in Omega:
        rot_mat = np.array([[np.cos(omega_j),
                             np.sin(omega_j), 0],
                            [-np.sin(omega_j),
                             np.cos(omega_j), 0], [0, 0, 1]])
        new_pt = np.zeros_like(pts_inst)
        new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
        if pts_inst.shape[1] > 3:
            new_pt[:, 3:] = pts_inst[:, 3:]
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
        inst_copy.append(inst_inst)
        sig_copy.append(sig_inst)
    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    inst_copy = np.concatenate(inst_copy, axis=0)
    sig_copy = np.concatenate(sig_copy, axis=0)
    return pts_copy, labels_copy, inst_copy, sig_copy

# def polarmix(pts1, labels1, inst1, sig1, pts2, labels2, inst2, sig2, alpha, beta, instance_classes, Omega):
#     # swapping
#     if np.random.random() < 0.5:
#         pts1, pts2, labels1, labels2, inst1, inst2, sig1, sig2 = swap_with_instance(
#             pts1, pts2, alpha, beta, labels1, labels2, inst1, inst2, sig1, sig2
#         )

#     # rotate-pasting
#     if np.random.random() < 1.0:
#         pts_copy, labels_copy, inst_copy, sig_copy = rotate_copy_with_instance(pts2, labels2, inst2, sig2, instance_classes, Omega)
#         if pts_copy is not None:
#             pts1 = np.concatenate((pts1, pts_copy), axis=0)
#             labels1 = np.concatenate((labels1, labels_copy), axis=0)
#             inst1 = np.concatenate((inst1, inst_copy), axis=0)
#             sig1 = np.concatenate((, sig_copy), axis=0)

#     return pts1, labels1, inst1, sig1


def polarmix(pts1, labels1, inst1, sig1, pts2, labels2, inst2, sig2, alpha, beta, instance_classes, Omega, target_classes=None):
    """
    只对 target_classes 指定的类别做 polarmix，其余类别不做增强。
    target_classes: list/set/None，类别id。如果为None或空，则所有类别都增强。
    """
    # 如果target_classes为空或None，则所有类别都增强
    if not target_classes:
        mask1 = np.ones_like(labels1, dtype=bool)
        mask2 = np.ones_like(labels2, dtype=bool)
    else:
        mask1 = np.isin(labels1, target_classes)
        mask2 = np.isin(labels2, target_classes)

    mask1 = mask1.reshape(-1)
    mask2 = mask2.reshape(-1)
    # 分别处理目标类别和非目标类别
    # 保证所有输入都是一维
    pts1_target, labels1_target, inst1_target, sig1_target = pts1[mask1], labels1[mask1], inst1[mask1], sig1[mask1]
    pts2_target, labels2_target, inst2_target, sig2_target = pts2[mask2], labels2[mask2], inst2[mask2], sig2[mask2]
    pts1_rest, labels1_rest, inst1_rest, sig1_rest = pts1[~mask1], labels1[~mask1], inst1[~mask1], sig1[~mask1]
    
    # 对目标类别做polarmix
    if np.random.random() < 0.5:
        pts1_target, pts2_target, labels1_target, labels2_target, inst1_target, inst2_target, sig1_target, sig2_target = swap_with_instance(
            pts1_target, pts2_target, alpha, beta, labels1_target, labels2_target, inst1_target, inst2_target, sig1_target, sig2_target
        )

    if np.random.random() < 1.0:
        pts_copy, labels_copy, inst_copy, sig_copy = rotate_copy_with_instance(
            pts2_target, labels2_target, inst2_target, sig2_target, instance_classes, Omega)
        if pts_copy is not None:
            pts1_target = np.concatenate((pts1_target, pts_copy), axis=0)
            labels1_target = np.concatenate((labels1_target, labels_copy), axis=0)
            inst1_target = np.concatenate((inst1_target, inst_copy), axis=0)
            sig1_target = np.concatenate((sig1_target, sig_copy), axis=0)

    
    sig1_target = np.asarray(sig1_target).reshape(-1, 1)
    sig1_rest = np.asarray(sig1_rest).reshape(-1, 1)
    
    # 合并目标类别和非目标类别
    pts1_out = np.concatenate([pts1_target, pts1_rest], axis=0)
    labels1_out = np.concatenate([labels1_target, labels1_rest], axis=0)
    inst1_out = np.concatenate([inst1_target, inst1_rest], axis=0)
    sig1_out = np.concatenate([sig1_target, sig1_rest], axis=0)

    return pts1_out, labels1_out, inst1_out, sig1_out