"""
三角化模块

实现新3D点的三角化，包括角度过滤和负深度过滤
"""
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Set

from config import SFMConfig
from core.scene import SceneManager, View, Point3D
from storage.database import Database


class Triangulator:
    """三角化器
    
    负责三角化新的3D点
    """
    
    def __init__(self, config: SFMConfig, db: Database):
        """初始化三角化器
        
        Args:
            config: SFM配置
        """
        self.config = config
        self.db = db
        self.logger = logging.getLogger("Triangulator")
    
    
    def triangulate_new_points(self, scene: SceneManager, image_id: int) -> int:
        """三角化新的3D点
        
        对于新注册的图像，与已注册图像之间的匹配点进行三角化
        
        Args:
            scene: 场景管理器
            image_id: 新注册的图像ID
            
        Returns:
            新三角化的点数量
        """
        self.logger.info(f"开始为图像 {image_id} 三角化新点")

        view = scene.views[image_id]
        if not view.is_registered:
            self.logger.error(f"图像 {image_id} 未注册")
            return 0

        registered_images = set(scene.get_registered_images())

        # 从match_graph获取有匹配的已注册图像，按匹配数量排序
        matched_with_count = []
        for other_id in list(scene.match_graph.get(image_id, {}).keys()):
            if other_id == image_id or other_id not in registered_images:
                continue
            try:
                matches = self.db.load_matches(image_id, other_id)
                matched_with_count.append((other_id, len(matches.matches)))
            except Exception:
                continue

        matched_with_count.sort(key=lambda x: x[1], reverse=True)

        new_points_count = 0
        for other_id, match_count in matched_with_count:
            self.logger.debug(f"三角化图像对 ({image_id}, {other_id})，匹配数: {match_count}")
            count = self._triangulate_image_pair(scene, image_id, other_id)
            new_points_count += count

        self.logger.info(f"为图像 {image_id} 三角化了 {new_points_count} 个新点")
        return new_points_count


    def _triangulate_image_pair(self, scene: SceneManager, id1: int, id2: int) -> int:
        """三角化一对图像之间的点
        
        Args:
            scene: 场景管理器
            id1: 第一个图像ID（load_matches的第一个参数，i对应id1）
            id2: 第二个图像ID（load_matches的第二个参数，j对应id2）
            
        Returns:
            新三角化的点数量
        """
        view1 = scene.views[id1]
        view2 = scene.views[id2]

        if not view1.is_registered or not view2.is_registered:
            return 0

        # 构建投影矩阵
        P1 = view1.K @ np.hstack((view1.R, view1.t))
        P2 = view2.K @ np.hstack((view2.R, view2.t))

        # 加载特征点和匹配（使用已有的self.db，不重复创建）
        try:
            features1 = self.db.load_keypoints(id1, load_desc=False)
            features2 = self.db.load_keypoints(id2, load_desc=False)
            matches = self.db.load_matches(id1, id2)
        except Exception as e:
            self.logger.error(f"加载特征点和匹配失败 ({id1}, {id2}): {e}")
            return 0

        if len(matches.matches) == 0:
            return 0

        # i → id1, j → id2（load_matches第一个参数对应i，第二个对应j）
        kp_indices1 = matches.matches[:, 0]
        kp_indices2 = matches.matches[:, 1]
        points1 = features1.keypoints_xy[kp_indices1]
        points2 = features2.keypoints_xy[kp_indices2]

        # 三角化
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        w = points_4d[3]
        valid_w = np.abs(w) > 1e-8
        points_3d = np.full((len(matches.matches), 3), np.nan, dtype=np.float32)
        points_3d[valid_w] = (points_4d[:3, valid_w] / w[valid_w]).T

        # 过滤：点在相机前方
        valid_front = self._check_point_in_front(points_3d, view1, view2)

        # 过滤：三角化基线角度足够
        valid_angle = self._check_triangulation_angle(points1, points2, view1, view2)

        # 过滤：重投影误差
        valid_reproj = self._check_reprojection_error(
            points_3d, points1, points2, view1, view2
        )

        valid_mask = valid_front & valid_angle & valid_reproj

        # 用obs_index快速查重，O(1)而不是O(N)
        new_points_count = 0
        for i in np.where(valid_mask)[0]:
            kp_idx1 = int(kp_indices1[i])
            kp_idx2 = int(kp_indices2[i])

            # 检查这两个特征点是否已经对应到3D点（用obs_index，O(1)）
            already_exists = (
                kp_idx1 in scene.obs_index.get(id1, {}) or
                kp_idx2 in scene.obs_index.get(id2, {})
            )
            if already_exists:
                continue

            observations = {id1: kp_idx1, id2: kp_idx2}
            scene.add_point3d(points_3d[i], observations)
            new_points_count += 1

        return new_points_count


    def _check_reprojection_error(
        self,
        points_3d: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray,
        camera1,
        camera2,
        max_error: float = 4.0
    ) -> np.ndarray:
        """检查重投影误差是否在阈值内
        
        Args:
            points_3d: Nx3 三角化点
            points1: Nx2 图像1中的2D点
            points2: Nx2 图像2中的2D点
            camera1: 相机1
            camera2: 相机2
            max_error: 最大允许重投影误差（像素）
            
        Returns:
            bool数组，True表示重投影误差合格
        """
        n = len(points_3d)
        valid = np.zeros(n, dtype=bool)

        for i in range(n):
            if np.any(np.isnan(points_3d[i])):
                continue

            pt = points_3d[i].reshape(3, 1)

            # 图像1重投影
            proj1 = camera1.K @ (camera1.R @ pt + camera1.t)
            proj1 = (proj1[:2] / proj1[2]).flatten()
            err1 = np.linalg.norm(proj1 - points1[i])

            # 图像2重投影
            proj2 = camera2.K @ (camera2.R @ pt + camera2.t)
            proj2 = (proj2[:2] / proj2[2]).flatten()
            err2 = np.linalg.norm(proj2 - points2[i])

            valid[i] = (err1 < max_error) and (err2 < max_error)

        return valid
    

    def _check_point_in_front(self, points_3d: np.ndarray, view1: View, 
                            view2: View) -> np.ndarray:
        """检查点是否在相机前方
        
        Args:
            points_3d: 3D点坐标 [N, 3]
            camera1: 第一个相机
            camera2: 第二个相机
            
        Returns:
            布尔掩码 [N]，表示点是否在两个相机前方
        """
        # 计算点在相机1坐标系中的坐标
        points_in_cam1 = np.dot(points_3d - view1.t.T, view2.R)
        
        # 计算点在相机2坐标系中的坐标
        points_in_cam2 = np.dot(points_3d - view1.t.T, view2.R)
        
        # 检查Z坐标是否为正
        valid1 = points_in_cam1[:, 2] > 0
        valid2 = points_in_cam2[:, 2] > 0
        
        return np.logical_and(valid1, valid2)
    

    def _check_triangulation_angle(self, points1: np.ndarray, points2: np.ndarray,
                                view1: View, view2: View) -> np.ndarray:
        """检查三角化角度是否在有效范围内
        
        Args:
            points1: 第一个图像中的点坐标 [N, 2]
            points2: 第二个图像中的点坐标 [N, 2]
            camera1: 第一个相机
            camera2: 第二个相机
            
        Returns:
            布尔掩码 [N]，表示三角化角度是否有效
        """
        # 计算相机中心
        center1 = view1.center
        center2 = view2.center
        
        # 计算从相机中心到归一化图像平面的射线
        rays1 = []
        rays2 = []
        
        for i in range(len(points1)):
            # 将像素坐标转换为归一化坐标
            p1 = np.array([points1[i, 0], points1[i, 1], 1.0])
            p2 = np.array([points2[i, 0], points2[i, 1], 1.0])
            
            # 应用相机内参的逆变换
            ray1 = np.dot(np.linalg.inv(view1.K), p1)
            ray2 = np.dot(np.linalg.inv(view2.K), p2)
            
            # 旋转到世界坐标系
            ray1 = np.dot(view1.R.T, ray1)
            ray2 = np.dot(view2.R.T, ray2)
            
            # 归一化
            ray1 = ray1 / np.linalg.norm(ray1)
            ray2 = ray2 / np.linalg.norm(ray2)
            
            rays1.append(ray1)
            rays2.append(ray2)
        
        rays1 = np.array(rays1)
        rays2 = np.array(rays2)
        
        # 计算射线之间的夹角
        cos_angles = np.sum(rays1 * rays2, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)  # 防止数值误差
        angles = np.arccos(cos_angles) * 180.0 / np.pi  # 转换为度
        
        # 检查角度是否在有效范围内
        min_angle = self.config.min_triangulation_angle_deg
        max_angle = self.config.max_triangulation_angle_deg
        
        return np.logical_and(angles >= min_angle, angles <= max_angle)