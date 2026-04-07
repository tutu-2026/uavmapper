"""
特征提取模块基类
"""
from typing import List, Dict
from abc import ABC, abstractmethod
from core.data_models import ImageMeta, FeatureData


class BaseFeatureExtractor(ABC):
    """特征提取器基类，所有特征提取器都应继承此类"""

    @abstractmethod
    def extract(self, image_meta: ImageMeta) -> FeatureData:
        """从单张图像中提取特征
        Args:
            image_meta: 图像元数据
        Returns:
            提取的特征数据
        """
        pass

    @abstractmethod
    def extract_batch(self, image_metas: List[ImageMeta]) -> Dict[int, FeatureData]:
        """批量提取特征
        Args:
            image_metas: 图像元数据列表
        Returns:
            {image_id: FeatureData} 字典
        """
        pass