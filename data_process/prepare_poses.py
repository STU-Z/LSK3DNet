import numpy as np
import os

def split_poses(pose_path, out_dir):
    # 先统计总行数
    with open(pose_path, 'r') as f:
        total = sum(1 for _ in f)
    os.makedirs(out_dir, exist_ok=True)
    with open(pose_path, 'r') as f:
        for idx, line in enumerate(f):
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))  # 变成4x4齐次矩阵
            out_path = os.path.join(out_dir, f"{idx:06d}.npy")
            np.save(out_path, T)
            # 简单进度条
            print(f"\rProgress: {idx+1}/{total} ({(idx+1)/total*100:.2f}%)", end='')
    print("\nAll poses saved.")

if __name__ == "__main__":
    sequences = [f"{i:02d}" for i in range(11)]  # 00~10
    base_dir = "/media/zrb/Zrb-TB2/kitti/SemanticKITTI_Data/SemanticKitti/sequences"
    for seq in sequences:
        pose_path = os.path.join(base_dir, seq, "poses.txt")
        out_dir = os.path.join(base_dir, seq, "poses_split")
        if os.path.exists(pose_path):
            print(f"\nProcessing sequence {seq} ...")
            split_poses(pose_path, out_dir)
        else:
            print(f"\nposes.txt not found for sequence {seq}, skipped.")