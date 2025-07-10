'''
Author: Zhangrunbang 254616730@qq.com
Date: 2025-06-26 09:30:30
LastEditors: Zhangrunbang 254616730@qq.com
LastEditTime: 2025-06-27 11:44:17
FilePath: /LSK3DNet/config/ratio_cal.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np

label=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
ratio=[0.031501833,0.040818519255974316,0.00016609538710764618,
       0.00039838616015114444,0.0020633612104619787,0.001649698,
       0.00017698551338515307 ,1.1065903904919655e-08,5.532951952459828e-09,0.19879647126983286,
       0.014717169549888214 ,0.14392298360372,0.0039048553037472045,0.1326861944777486,
       0.0723592229456223,0.26681502148037506 ,0.006035012012626033,0.07814222006271769,
       0.002855498193863172,0.0006155958086189918]
seg_labelweights=[0,55437630,320797,541736,2578735,3274484,552662,184064,78858,240942562,17294618,
                  170599734,6369672,230413074,101130274,476491114,9833174,129609852,4506626,1168181]




'''
用类别反比权重（inverse frequency weighting）来平衡类别不均衡问题。常见做法是：
权重 = 1 / (类别比例 + ε)，再归一化。
输出即为每个类别的损失权重，可直接用于 loss 函数的 class_weight 参数。

说明：

权重越大，说明该类别越稀有，训练时损失会被放大，模型更关注小样本类别。
归一化后不会影响 loss 的绝对量级。
'''

# 防止除零
epsilon = 1e-6
ratio = np.array(ratio)
weights = 1.0 / (ratio + epsilon)
# 归一化，使权重和为类别数
weights = weights * len(weights) / weights.sum()

for l, w in zip(label, weights):
    print(f"label {l}: weight {w:.8f}")
    
seg_labelweights=np.array(seg_labelweights)
total = seg_labelweights.sum()
ratios = seg_labelweights / total
for l, r in zip(label, ratios):
    print(f"label {l}: ratio {r:.8f}")
seg_labelweights_final = 1 / (ratios + 0.02)  
for l, w in zip(label, seg_labelweights_final):
    print(f"label {l}: seg_labelweights_final {w:.8f}")
