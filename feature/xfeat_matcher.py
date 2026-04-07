"""
基于 XFeat 的特征匹配器

使用 XFeat 模型进行特征匹配，支持多进程加速
"""
import torch
import faiss
import numpy as np
from typing import Tuple

from feature.base_matcher import BaseMatcher
from config import FeatureConfig
from storage.database import Database


class XFeatMatcher(BaseMatcher):
    """基于 XFeat 的特征匹配器
    
    使用 XFeat 模型进行特征描述符匹配，支持多进程加速。
    XFeat 是一种高效的特征提取和匹配模型，适用于大规模图像匹配。
    """

    def __init__(self, db: Database, config: FeatureConfig):
        """初始化 XFeat 特征匹配器
        
        Args:
            db: 数据库实例，用于加载和保存特征
            config: 特征配置，包含匹配参数
        """
        self.num_processes = config.batch_size
        self.ratio_threshold = config.lowes_ratio
        super().__init__(db, config)


    @staticmethod
    def _init_worker():
        """初始化工作进程，加载 XFeat 模型"""
        global global_xfeat
        global_xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, force_reload=False, skip_validation=True)
        print("XFeat 模型已在工作进程中加载")


    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用 XFeat 模型匹配两组描述符
        
        Args:
            desc1: 查询描述符 [N, 128]
            desc2: 参考描述符 [M, 128]
            
        Returns:
            (matches, scores) 元组
            matches: [N'] 数组，表示desc1中每个描述符在desc2中的最近邻索引
            scores: [N'] 数组，表示最近邻的距离比率（第一近邻/第二近邻）
        """
        # 确保描述符是float32类型
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)

        # 检查描述符数量
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        # 转换成PyTorch张量
        descriptors_tensor1 = torch.from_numpy(desc1)
        descriptors_tensor2 = torch.from_numpy(desc2)

        idxs1, idxs2 = global_xfeat.match(descriptors_tensor1, descriptors_tensor2)
        
        # 转换为numpy数组
        idxs1_np = idxs1.cpu().numpy()
        idxs2_np = idxs2.cpu().numpy()

        # 计算匹配质量评分
        scores = self._compute_match_scores(desc1, desc2, idxs1_np, idxs2_np)

        # 将索引合并为匹配对数组
        matches = np.column_stack((idxs1_np, idxs2_np))

        return matches, scores
    

    def _compute_match_scores(self, desc1: np.ndarray, desc2: np.ndarray, 
                          idxs1: np.ndarray, idxs2: np.ndarray) -> np.ndarray:
        """计算匹配质量评分 - 使用 FAISS 加速
        
        使用 FAISS 库进行高效的 KNN 搜索，计算每个匹配点的最近邻和次近邻距离比率
        
        Args:
            desc1: 第一组描述符
            desc2: 第二组描述符
            idxs1: desc1 中匹配点的索引
            idxs2: desc2 中匹配点的索引
            
        Returns:
            匹配质量评分数组，值越小表示匹配质量越高
        """
        if len(idxs1) == 0:
            return np.array([], dtype=np.float32)
        
        # 确保描述符是 float32 类型
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
        
        # 创建 FAISS 索引
        d = desc2.shape[1]  # 描述符维度
        index = faiss.IndexFlatL2(d)
        
        # 添加参考描述符
        index.add(desc2)
        
        # 对每个匹配点查询 2 个最近邻
        k = min(2, desc2.shape[0])
        distances, indices = index.search(desc1[idxs1], k)
        
        # 计算比率分数
        scores = np.ones(len(idxs1), dtype=np.float32)
        
        for i, (idx2, nn_indices) in enumerate(zip(idxs2, indices)):
            # 检查匹配点是否是最近邻
            if nn_indices[0] == idx2:
                # 匹配点是最近邻，使用最近邻和次近邻的距离比率
                if k > 1:
                    nearest_dist = distances[i, 0]
                    second_nearest_dist = distances[i, 1]
                    if second_nearest_dist > 0:
                        scores[i] = nearest_dist / second_nearest_dist
            else:
                # 匹配点不是最近邻，使用匹配点距离和最近邻距离的比率
                match_dist = np.sqrt(np.sum((desc1[idxs1[i]] - desc2[idx2]) ** 2))
                nearest_dist = distances[i, 0]
                if nearest_dist > 0:
                    scores[i] = match_dist / nearest_dist
        
        return scores