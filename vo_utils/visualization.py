import cv2
import numpy as np

from core.data_models import MatchData, FeatureData

def draw_feature_matches(
    img1_path: str, 
    img2_path: str, 
    keypoints1: FeatureData, 
    keypoints2: FeatureData, 
    matches: MatchData, 
    output_path: str = None,
    max_matches: int = 2000  # 最大绘制的匹配数量
) -> None:
    """
    可视化两张图像之间的特征点匹配关系，根据评分绘制最高分的前 max_matches 个匹配。

    Args:
        img1_path (str): 第一张图像的路径。
        img2_path (str): 第二张图像的路径。
        keypoints1 (FeatureData): 第一张图像的特征点数据。
        keypoints2 (FeatureData): 第二张图像的特征点数据。
        matches (MatchData): 特征点匹配数据，包含匹配对的索引和评分。
        output_path (str, optional): 如果提供，将保存结果图像到指定路径。
        max_matches (int, optional): 最大绘制的匹配数量，默认为 50。
    """
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(f"无法加载图像，请检查路径是否正确：{img1_path}, {img2_path}")
        return

    # 将图像拼接在一起
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    combined_img = np.zeros((height, width, 3), dtype=np.uint8)
    combined_img[:img1.shape[0], :img1.shape[1], :] = img1
    combined_img[:img2.shape[0], img1.shape[1]:, :] = img2

    # 根据评分对匹配对进行排序，选择评分最高的前 max_matches 个匹配
    sorted_indices = np.argsort(matches.scores)[::-1]  # 按评分从高到低排序
    top_matches = matches.matches[sorted_indices[:max_matches]]

    # 绘制匹配点
    for match in top_matches:
        index1, index2 = match
        pt1 = tuple(keypoints1.keypoints_xy[index1].astype(int))
        pt2 = tuple(keypoints2.keypoints_xy[index2].astype(int) + np.array([img1.shape[1], 0]))  # 偏移x坐标到第二张图像的区域
        color = tuple(np.random.randint(0, 255, size=3).tolist())  # 随机颜色
        cv2.circle(combined_img, pt1, 5, color, 2)
        cv2.circle(combined_img, pt2, 5, color, 2)
        cv2.line(combined_img, pt1, pt2, color, 3)

    # 显示图像
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.imshow(combined_img)
    plt.axis("off")
    plt.title(f"Top {len(top_matches)} Feature Matches")
    plt.show()

    # 保存到文件
    if output_path:
        combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, combined_img_bgr)
        print(f"匹配结果已保存到 {output_path}")