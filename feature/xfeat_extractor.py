"""
基于 XFeat 的特征提取器
"""
import cv2
import torch
import numpy as np
import multiprocessing
from typing import Dict, List
from functools import partial

from core.data_models import ImageMeta, FeatureData
from feature.base_extractor import BaseFeatureExtractor


class XFeatExtractor(BaseFeatureExtractor):
    """基于 XFeat 的特征提取器"""

    def __init__(self, top_k: int = 8000):
        """
        初始化 XFeat 提取器
        Args:
            top_k: 提取的最大特征点数量
        """
        self.top_k = top_k
        self.model = None  # 模型将在子进程中初始化


    @staticmethod
    def _init_worker():
        """初始化工作进程，加载 XFeat 模型"""
        global global_xfeat
        global_xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, force_reload=False, skip_validation=True,)
        print("XFeat 模型已在工作进程中加载")


    @staticmethod
    def _process_single_image(image_meta: ImageMeta, top_k: int) -> FeatureData:
        """从单张图像中提取特征（子进程调用）
        Args:
            image_meta: 图像元数据
            top_k: 提取的最大特征点数量
        Returns:
            提取的特征数据
        """
        global global_xfeat

        # 加载图像
        img = cv2.imread(image_meta.image_path)
        if img is None:
            print(f"无法读取图片: {image_meta.image_path}")
            return FeatureData(
                image_id=image_meta.image_id,
                num_keypoints=0,
                keypoints_xy=np.zeros((0, 2), dtype=np.float32),
                scales=np.zeros(0, dtype=np.float32),
                orientations=np.zeros(0, dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                descriptors=np.zeros((0, 128), dtype=np.float32),
            )

        # # 转换颜色空间
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(img.shape) == 3:
            img = img[None, ...]

        if isinstance(img, np.ndarray):
            img = torch.tensor(img).permute(0,3,1,2)/255

        # 提取特征
        output = global_xfeat.detectAndComputeDense(img, top_k=top_k)

        keypoints = output["keypoints"]
        descriptors = output["descriptors"]
        scales = output["scales"]

        if len(keypoints) == 0:
            return FeatureData(
                image_id=image_meta.image_id,
                num_keypoints=0,
                keypoints_xy=np.zeros((0, 2), dtype=np.float32),
                scales=np.zeros(0, dtype=np.float32),
                orientations=np.zeros(0, dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                descriptors=np.zeros((0, 128), dtype=np.float32),
            )

        # 提取关键点属性
        keypoints = keypoints.squeeze(0)
        keypoints_xy = np.array([[kp[0], kp[1]] for kp in keypoints], dtype=np.float32)
        scales= np.array(scales, dtype=np.float32)
        orientations = np.zeros(len(keypoints), dtype=np.float32) 
        scores = np.zeros(0, dtype=np.float32)

        return FeatureData(
            image_id=image_meta.image_id,
            num_keypoints=len(keypoints),
            keypoints_xy=keypoints_xy,
            scales=scales,
            orientations=orientations,
            scores=scores,
            descriptors=descriptors.cpu().numpy().squeeze(0),
        )


    def extract(self, image_meta: ImageMeta) -> FeatureData:
        """从单张图像中提取特征"""
        return self._process_single_image(image_meta, self.top_k)


    def extract_batch(self, image_metas: List[ImageMeta], num_processes: int = None) -> Dict[int, FeatureData]:
        """批量提取特征
        Args:
            image_metas: 图像元数据列表
            num_processes: 并行使用的进程数
        Returns:
            {image_id: FeatureData} 字典
        """
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        print(f"使用 {num_processes} 个进程进行并行处理")

        # 创建部分函数，固定参数
        process_func = partial(self._process_single_image, top_k=self.top_k)

        # 使用进程池并行处理
        with multiprocessing.Pool(processes=num_processes, initializer=self._init_worker) as pool: results = pool.map(process_func, image_metas)

        return {meta.image_id: result for meta, result in zip(image_metas, results)}