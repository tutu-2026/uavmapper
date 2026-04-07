import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, force_reload=False, skip_validation=True)

image1 = r"E:\uavData\new_test\images\DJI_20251016170631_0004_V.jpeg"
image2 = r"E:\uavData\new_test\images\DJI_20251016170637_0006_V.jpeg"

img1 = cv2.imread(image1)
img2 = cv2.imread(image2)

# 转换为RGB格式用于显示
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 获取匹配点
matches_list = xfeat.match_xfeat_star(img1, img2)
print(f"找到 {matches_list[0].shape[0]} 对匹配点")

# 生成不同的颜色
def generate_colors(n):
    """
    生成n个不同的颜色
    使用HSV颜色空间，H值均匀分布，S和V固定为较高的值
    """
    colors = []
    for i in range(n):
        # 在HSV空间中，H值范围是[0,180)，我们均匀分布H值
        h = int(180 * i / n) % 180
        s = 255  # 饱和度最大
        v = 255  # 亮度最大
        
        # 转换为BGR颜色
        bgr_color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0][0]
        # 转换为RGB元组
        rgb_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        colors.append(rgb_color)
    return colors

# 绘制匹配图
def draw_matches(img1, img2, kpts1, kpts2, matches_mask=None):
    """
    绘制两张图像之间的特征点匹配，每个匹配使用不同的颜色
    
    参数:
    img1, img2: 输入图像
    kpts1, kpts2: 两张图像中的关键点坐标
    matches_mask: 可选，用于选择性显示部分匹配
    """
    # 创建拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 创建输出图像
    height = max(h1, h2)
    width = w1 + w2
    output = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 将两张图像放入输出图像
    output[:h1, :w1] = img1
    output[:h2, w1:w1+w2] = img2
    
    # 随机选择一部分匹配点进行显示，避免图像过于拥挤
    if matches_mask is None:
        # 如果匹配点太多，只显示部分
        if len(kpts1) > 100:
            matches_mask = np.random.choice(len(kpts1), 100, replace=False)
        else:
            matches_mask = range(len(kpts1))
    
    # 生成不同的颜色
    colors = generate_colors(len(matches_mask))
    
    # 绘制匹配线
    for idx, i in enumerate(matches_mask):
        # 获取两个图像中对应的关键点
        pt1 = (int(kpts1[i][0]), int(kpts1[i][1]))
        pt2 = (int(kpts2[i][0]) + w1, int(kpts2[i][1]))
        
        # 使用不同颜色绘制线条
        color = colors[idx]
        
        # 绘制线条
        cv2.line(output, pt1, pt2, color, 2)
        # 在关键点位置绘制小圆圈，使用相同的颜色
        cv2.circle(output, pt1, 4, color, -1)
        cv2.circle(output, pt2, 4, color, -1)
    
    return output

# 从matches_list中获取关键点
kpts1 = matches_list[0]
kpts2 = matches_list[1]

# 随机选择100个匹配点进行显示
num_matches = len(kpts1)
if num_matches > 100:
    indices = np.random.choice(num_matches, 100, replace=False)
else:
    indices = range(num_matches)

# 绘制匹配图
match_img = draw_matches(img1_rgb, img2_rgb, kpts1, kpts2, indices)

# 显示结果
plt.figure(figsize=(20, 10))
plt.imshow(match_img)
plt.title(f'特征点匹配结果 (显示 {len(indices)}/{num_matches} 对匹配点)')
plt.axis('off')
plt.tight_layout()
plt.savefig('feature_matches.png', dpi=300)
plt.show()

# 也可以使用OpenCV直接显示
# cv2.imshow('Feature Matches', cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 打印匹配点数量
print(f"总共找到 {num_matches} 对匹配点")