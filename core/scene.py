"""
场景管理模块

定义SFM重建过程中的运行时场景状态
"""
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from storage.database import Database

@dataclass
class View:
    """视图类
    
    存储相机的位姿和内参
    """
    image_id: int
    K: np.ndarray                               # 3×3 相机内参矩阵
    R: np.ndarray                               # 3×3 旋转矩阵（世界系 → 相机系）
    t: np.ndarray                               # 3×1 平移向量
    gps_local_xyz: Optional[np.ndarray] = None  # GPS转换后的局部ENU坐标（米）
    prior_weight: float = 1.0                   # BA先验权重
    is_registered: bool = False
    
    @property
    def center(self) -> np.ndarray:
        """计算相机中心在世界坐标系中的位置
        
        Returns:
            相机中心的3D坐标
        """
        # C = -R^T * t
        return -np.dot(self.R.T, self.t).flatten()
    
    @property
    def P(self) -> np.ndarray:
        """计算投影矩阵 P = K[R|t]
        
        Returns:
            3x4投影矩阵
        """
        Rt = np.column_stack((self.R, self.t))
        return np.dot(self.K, Rt)

    @property
    def has_prior(self) -> bool:
        return self.prior_position is not None


@dataclass
class Point3D:
    """三维点类
    
    存储重建的3D点及其观测信息
    """
    point_id: int
    xyz: np.ndarray                                             # 世界坐标系下的3D位置
    observations: Dict[int, int] = field(default_factory=dict)  # {image_id: keypoint_index}
    is_outlier: bool = False
    
    @property
    def track_length(self) -> int:
        """获取轨迹长度（观测数量）
        
        Returns:
            观测该点的图像数量
        """
        return len(self.observations)


class SceneManager:
    """场景管理器
    
    管理SFM重建过程中的视图和3D点
    """
    
    def __init__(self):
        """初始化场景管理器"""
        self.views: Dict[int, View] = {}
        self.points3d: Dict[int, Point3D] = {}
        self.next_point_id: int = 0
        # 图像匹配关系：{image_id: {other_image_id: 匹配点数量}}
        self.match_graph: Dict[int, Dict[int, int]] = {}
        # 图像共视关系: {image_id: {other_image_id: 共视点数量}}
        self.covisibility_graph = defaultdict(lambda: defaultdict(int))
        # 图像-点云索引: {image_id: {point3d_id, ...}}
        self.image_points_index: Dict[int, Set[int]] = {}
        # {image_id: {kp_idx: point3d_id}}
        self.obs_index = defaultdict(dict)


    def init_match_graph(self, db: 'Database') -> None:
        """从数据库初始化图像匹配关系图
        
        Args:
            db: 数据库实例
        """
        # 初始化匹配图
        self.match_graph = defaultdict(dict)

        # 加载所有匹配对
        try:
            pairs = db.load_all_pairs()
            
            # 对每个匹配对，加载匹配数据并更新匹配图
            for id1, id2 in pairs:
                try:
                    # 优先使用几何验证后的内点匹配
                    matches = db.load_inlier_matches(id1, id2)
                    if matches is None:
                        matches = db.load_matches(id1, id2)
                    
                    if matches is not None:
                        match_count = len(matches.matches)
                        
                        # 更新匹配图（双向）
                        self.match_graph[id1][id2] = match_count
                        self.match_graph[id2][id1] = match_count
                except Exception:
                    # 忽略加载匹配失败的情况
                    pass
        except Exception as e:
            # 如果加载匹配对失败，记录错误但继续执行
            print(f"加载匹配对时出错: {str(e)}")


    def add_view(self, view: View) -> None:
        """添加相机到场景
        
        Args:
            camera: 相机对象
        """
        self.views[view.image_id] = view
        self.image_points_index[view.image_id] = set()
    

    def add_point3d(self, xyz: np.ndarray, observations: Dict[int, int]) -> int:
        """添加3D点到场景
        
        Args:
            xyz: 3D点坐标
            observations: 观测信息 {image_id: keypoint_index}
            
        Returns:
            新创建的点ID
        """
        point_id = self.next_point_id
        self.next_point_id += 1
        
        point = Point3D(point_id=point_id, xyz=xyz, observations=observations)
        self.points3d[point_id] = point
        
        # 更新图像-点云索引
        for image_id, kp_idx in observations.items():  # ← 修复 bug
            if image_id in self.image_points_index:
                self.image_points_index[image_id].add(point_id)
            self.obs_index[image_id][kp_idx] = point_id  # ← 移到if外面，确保都更新

        # 更新共视关系
        self._update_covisibility(point)
        
        return point_id
    

    def remove_point3d(self, point_id: int) -> None:
        """从场景中移除3D点
        
        Args:
            point_id: 要移除的点ID
        """
        if point_id not in self.points3d:
            return
            
        point = self.points3d[point_id]
        
        # 更新图像-点云索引
        for image_id in point.observations:
            if image_id in self.image_points_index:
                self.image_points_index[image_id].discard(point_id)
        
        # 更新共视关系（减少共视点数）
        image_ids = list(point.observations.keys())
        for i in range(len(image_ids)):
            for j in range(i+1, len(image_ids)):
                id1, id2 = image_ids[i], image_ids[j]
                if id1 in self.covisibility_graph and id2 in self.covisibility_graph[id1]:
                    self.covisibility_graph[id1][id2] -= 1
                    self.covisibility_graph[id2][id1] -= 1
        
        # 删除点
        del self.points3d[point_id]
    

    def _update_covisibility(self, point: Point3D) -> None:
        image_ids = list(point.observations.keys())
        for i in range(len(image_ids)):
            for j in range(i + 1, len(image_ids)):
                id1, id2 = image_ids[i], image_ids[j]
                self.covisibility_graph[id1][id2] += 1
                self.covisibility_graph[id2][id1] += 1
    

    def get_covisibility(self, image_id: int) -> Dict[int, int]:
        """获取图像的共视关系
        
        Args:
            image_id: 图像ID
            
        Returns:
            {other_image_id: 共视点数量}
        """
        if image_id not in self.covisibility_graph:
            return {}
        return self.covisibility_graph[image_id]
    

    def get_points_observed_by(self, image_id: int) -> Dict[int, Point3D]:
        """获取图像观测到的所有3D点
        
        Args:
            image_id: 图像ID
            
        Returns:
            {point_id: Point3D对象}
        """
        if image_id not in self.image_points_index:
            return {}
        
        result = {}
        for point_id in self.image_points_index[image_id]:
            if point_id in self.points3d:
                result[point_id] = self.points3d[point_id]
        
        return result
    

    def get_registered_images(self) -> List[int]:
        """获取所有已注册的图像ID
        
        Returns:
            已注册的图像ID列表
        """
        return [image_id for image_id, view in self.views.items() if view.is_registered]
    

    def get_statistics(self) -> Dict[str, int]:
        """获取场景统计信息
        
        Returns:
            包含统计信息的字典
        """
        registered_count = sum(1 for view in self.views.values() if view.is_registered)
        valid_points = sum(1 for point in self.points3d.values() if not point.is_outlier)
        
        return {
            "total_cameras": len(self.views),
            "registered_cameras": registered_count,
            "total_points": len(self.points3d),
            "valid_points": valid_points
        }