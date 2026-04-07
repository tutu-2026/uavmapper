"""
场景图管理模块

维护图像之间的共视关系图，用于增量SFM中的图像选择和局部BA
"""
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter

from core.scene import SceneManager


class SceneGraph:
    """场景图类
    
    管理图像之间的共视关系，用于增量SFM中的图像选择和局部BA
    """
    
    def __init__(self, scene: SceneManager):
        """初始化场景图
        
        Args:
            scene: 场景管理器
        """
        self.scene = scene
    

    def get_image_covisibility_score(self, image_id: int, registered_images: Set[int]) -> Dict[int, float]:
        """计算一张图像与已注册图像的共视分数
        
        Args:
            image_id: 目标图像ID
            registered_images: 已注册图像ID集合
            
        Returns:
            {registered_image_id: 共视分数}，分数越高表示共视关系越强
        """
        # 获取图像的共视关系
        covisibility = self.scene.get_covisibility(image_id)
        
        # 计算与已注册图像的共视分数
        scores = {}
        for reg_id in registered_images:
            if reg_id in covisibility:
                # 共视点数量作为基础分数
                scores[reg_id] = covisibility[reg_id]
        
        return scores
    

    def find_local_bundle_window(self, image_id: int, window_size: int) -> List[int]:
        """查找用于局部BA的窗口
        
        基于共视关系，选择与目标图像共视最强的N个图像
        
        Args:
            image_id: 中心图像ID
            window_size: 窗口大小
            
        Returns:
            选定的图像ID列表，包括中心图像
        """
        # 获取图像的共视关系
        covisibility = self.scene.get_covisibility(image_id)
        
        # 按共视点数量排序
        sorted_covis = sorted(covisibility.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前N-1个共视最强的图像
        selected = [image_id]  # 首先包含中心图像
        for other_id, _ in sorted_covis:
            if len(selected) >= window_size:
                break
            if self.scene.views.get(other_id, None) is not None and self.scene.views[other_id].is_registered:
                selected.append(other_id)
        
        return selected
    

    def find_fixed_images(self, image_ids: List[int], num_fixed: int) -> List[int]:
        # 按 scene.views 中的顺序排序（即注册顺序）
        view_order = list(self.scene.views.keys())  # [0, 2, 1, ...]
        sorted_ids = sorted(image_ids, key=lambda x: view_order.index(x))
        
        max_fixed = max(1, len(sorted_ids) - 1)
        actual_fixed = min(num_fixed, max_fixed)
        return sorted_ids[:actual_fixed]
    

    def find_next_images_to_register(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """查找下一批待注册的图像
        
        基于与已注册图像的共视强度，选择最有希望成功注册的图像
        
        Args:
            top_k: 返回的候选图像数量
            
        Returns:
            [(image_id, score), ...]，按分数降序排列
        """
        registered_images = set(self.scene.get_registered_images())
        if not registered_images:
            return []
        
        candidates = {}  # {image_id: 匹配点总数}
        # 遍历所有已注册图像
        for reg_id in registered_images:
            # 获取与已注册图像有匹配关系的其他图像
            # 这里使用match_graph而不是covisibility_graph
            matches = self.scene.match_graph.get(reg_id, {})
            
            # 对于每个有匹配关系的图像
            for other_id, match_count in matches.items():
                # 如果是未注册图像
                if other_id not in registered_images:
                    # 累加匹配点数量
                    if other_id not in candidates:
                        candidates[other_id] = 0
                    candidates[other_id] += match_count
            
        # 按匹配点数量降序排列
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_candidates[:top_k]
    

    def compute_track_statistics(self) -> Dict[str, float]:
        """计算场景中的轨迹统计信息
        
        Returns:
            包含统计信息的字典
        """
        if not self.scene.points3d:
            return {"avg_track_length": 0, "max_track_length": 0, "min_track_length": 0}
        
        track_lengths = [point.track_length for point in self.scene.points3d.values() 
                        if not point.is_outlier]
        
        if not track_lengths:
            return {"avg_track_length": 0, "max_track_length": 0, "min_track_length": 0}
        
        return {
            "avg_track_length": np.mean(track_lengths),
            "max_track_length": np.max(track_lengths),
            "min_track_length": np.min(track_lengths)
        }
    

    def compute_image_connectivity(self) -> Dict[int, int]:
        """计算每张图像的连接度
        
        连接度定义为与该图像有共视关系的其他图像数量
        
        Returns:
            {image_id: 连接度}
        """
        connectivity = {}
        for image_id in self.scene.cameras:
            covis = self.scene.get_covisibility(image_id)
            connectivity[image_id] = len(covis)
        
        return connectivity