"""
增量注册模块

实现增量式SFM的主循环，注册新图像到现有重建中
"""
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Set
import heapq

from config import SFMConfig
from core.scene import SceneManager, View, Point3D
from core.scene_graph import SceneGraph
from storage.database import Database
from sfm.triangulator import Triangulator
from core.geo_utils import GPSLocalConverter

class IncrementalRegistrar:
    """增量式SFM注册器
    
    实现增量式SFM的主循环，注册新图像到现有重建中
    """
    
    def __init__(self, db: Database, scene: SceneManager, 
                triangulator: Triangulator, config: SFMConfig):
        """初始化增量注册器
        
        Args:
            db: 数据库实例
            scene: 场景管理器
            triangulator: 三角化器
            config: SFM配置
        """
        self.db = db
        self.scene = scene
        self.triangulator = triangulator
        self.config = config
        self.logger = logging.getLogger("IncrementalRegistrar")
        
        # 初始化图像匹配关系图
        self.scene.init_match_graph(db)

        # 场景图，用于管理图像共视关系
        self.scene_graph = SceneGraph(scene)
        
        # 已注册和未注册的图像ID集合
        self.registered_images = set()
        self.unregistered_images = set()
        
        # 初始化图像集合
        self._init_image_sets()
    
    
    def _init_image_sets(self) -> None:
        """初始化已注册和未注册的图像ID集合"""
        # 加载所有图像
        all_images = self.db.load_all_images()
        all_image_ids = {img.image_id for img in all_images}
        
        # 已注册的图像
        self.registered_images = set(self.scene.get_registered_images())
        
        # 未注册的图像
        self.unregistered_images = all_image_ids - self.registered_images
        
        self.logger.info(f"已注册图像: {len(self.registered_images)}, 未注册图像: {len(self.unregistered_images)}")
    

    def find_next_best_images(self) -> List[int]:
        """查找下一批最佳待注册图像（批量注册）"""
        if not self.registered_images or not self.unregistered_images:
            return []
        if not self.scene.points3d:
            return []

        threshold_ratio = 0.75          # OpenMVG 默认阈值
        min_correspondences = 30        # 最少 2D-3D 对应数，低于此直接排除
        correspondences = {}            # {image_id: count}

        for reg_id in self.registered_images:
            matched_image_ids = self.scene.match_graph.get(reg_id, {}).keys()

            for image_id in matched_image_ids:
                if image_id not in self.unregistered_images:
                    continue

                matches = self.db.load_matches(image_id, reg_id)
                if matches is None:
                    continue

                obs_map = self.scene.obs_index.get(reg_id, {})
                if not obs_map:
                    continue

                for i, j in matches.matches:
                    kp_idx_reg = j if image_id < reg_id else i
                    if kp_idx_reg in obs_map:
                        correspondences[image_id] = correspondences.get(image_id, 0) + 1

        if not correspondences:
            return []

        # 过滤掉 correspondences 太少的图像
        correspondences = {
            k: v for k, v in correspondences.items() 
            if v >= min_correspondences
        }
        if not correspondences:
            return []

        candidates = sorted(correspondences.items(), key=lambda x: x[1], reverse=True)
        best_count = candidates[0][1]
        threshold = int(threshold_ratio * best_count)

        # 返回所有达到阈值的图像
        return [image_id for image_id, count in candidates if count >= threshold]

    
    def register_next_image(self, image_id: int) -> bool:
        """注册下一个图像到重建中
        
        Args:
            image_id: 要注册的图像ID
            
        Returns:
            是否成功注册
        """
        if image_id in self.registered_images:
            self.logger.warning(f"图像 {image_id} 已经注册")
            return False
        
        self.logger.info(f"尝试注册图像 {image_id}")
        
        # 1. 加载图像元数据
        try:
            meta = self.db.load_image_meta(image_id)
        except Exception as e:
            self.logger.error(f"加载图像 {image_id} 元数据失败: {str(e)}")
            return False
        
        # 2. 查找2D-3D对应关系
        correspondences_2d3d = self._find_2d3d_correspondences(image_id)
        
        if len(correspondences_2d3d) < self.config.min_pnp_inliers:
            self.logger.warning(f"图像 {image_id} 的2D-3D对应关系不足，无法注册")
            return False
        
        # 3. 使用PnP求解相机位姿
        success, R, t, inliers = self._solve_pnp(meta, correspondences_2d3d)
        
        if not success:
            self.logger.warning(f"图像 {image_id} 的PnP求解失败")
            return False
        
        # 建立gps转换器，把gps坐标转换为ENU坐标
        local_xyz = GPSLocalConverter.to_local_enu(meta.gps_lat, meta.gps_lon, meta.gps_alt)

        # 4. 创建相机对象并添加到场景
        view = View(
            image_id=image_id,
            K=meta.K,
            R=R,
            t=t,
            gps_local_xyz=local_xyz,
            is_registered=True
        )
        
        self.scene.add_view(view)
        
        # 5. 更新已注册和未注册的图像集合
        self.registered_images.add(image_id)
        self.unregistered_images.discard(image_id)  # 不存在也不报错
        
        # 6. 三角化新的3D点
        self._triangulate_new_points(image_id)
        
        # 7. 过滤异常点
        self._filter_outliers(image_id)
        
        self.logger.info(f"成功注册图像 {image_id}，内点数: {len(inliers)}")
        return True
    

    def _find_2d3d_correspondences(self, image_id: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        # 1. 加载特征点
        try:
            features = self.db.load_keypoints(image_id, load_desc=False)
        except Exception as e:
            self.logger.error(f"加载图像 {image_id} 特征失败: {e}")
            return []

        correspondences = []
        seen_point3d_ids = set()  # 避免同一个3D点被多次匹配添加

        for reg_id in self.registered_images:
            try:
                matches = self.db.load_matches(image_id, reg_id)
            except Exception:
                continue

            for i, j in matches.matches:
                # 确定当前图像和已注册图像各自的特征点索引, i永远对应 image_id（当前图），j 永远对应 reg_id（已注册图）
                kp_idx_curr = i
                kp_idx_reg  = j

                # 直接用反向索引查找，O(1)
                point3d_id = self.scene.obs_index.get(reg_id, {}).get(kp_idx_reg)
                if point3d_id is None:
                    continue

                # 避免重复添加同一个3D点
                if point3d_id in seen_point3d_ids:
                    continue
                seen_point3d_ids.add(point3d_id)

                point_2d = features.keypoints_xy[kp_idx_curr]
                point_3d = self.scene.points3d[point3d_id].xyz
                correspondences.append((point_2d, point_3d))

        return correspondences

    

    def _solve_pnp(self, meta, correspondences_2d3d):
        if len(correspondences_2d3d) < 4:
            return False, None, None, []

        points_2d = np.array([corr[0] for corr in correspondences_2d3d], dtype=np.float64)
        points_3d = np.array([corr[1] for corr in correspondences_2d3d], dtype=np.float64)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, meta.K,
                distCoeffs=meta.dist_coeffs,
                iterationsCount=1000,
                reprojectionError=self.config.pnp_ransac_threshold,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success or inliers is None or len(inliers) < self.config.min_pnp_inliers:
                return False, None, None, []

            inlier_idx = inliers.flatten()
            inlier_ratio = len(inlier_idx) / len(points_2d)
            if inlier_ratio < self.config.min_pnp_inlier_ratio:
                return False, None, None, []

            # ── 用内点做LM非线性精化 ──────────────────────────────
            inlier_pts3d = points_3d[inlier_idx]
            inlier_pts2d = points_2d[inlier_idx]
            cv2.solvePnPRefineLM(
                inlier_pts3d, inlier_pts2d,
                meta.K, meta.dist_coeffs,
                rvec, tvec  # 原地修改
            )

            R, _ = cv2.Rodrigues(rvec)
            return True, R, tvec, inlier_idx.tolist()

        except Exception as e:
            self.logger.error(f"PnP求解出错: {str(e)}")
            return False, None, None, []

    

    def _triangulate_new_points(self, image_id: int) -> None:
        """三角化新的3D点
        
        Args:
            image_id: 新注册的图像ID
        """
        # 使用三角化器添加新的3D点
        self.triangulator.triangulate_new_points(self.scene, image_id)
    

    def _filter_outliers(self, image_id: int) -> None:
        """过滤异常点
        
        移除重投影误差过大的点
        
        Args:
            image_id: 图像ID
        """
        view = self.scene.views[image_id]
        
        # 获取图像观测到的所有3D点
        observed_points = self.scene.get_points_observed_by(image_id)
        
        # 计算投影矩阵
        P = view.P
        
        # 检查每个点的重投影误差
        outliers = []
        
        for point_id, point in observed_points.items():
            # 获取3D点坐标
            point_3d = np.append(point.xyz, 1.0)  # 转换为齐次坐标
            
            # 投影到图像平面
            projected = np.dot(P, point_3d)
            projected = projected[:2] / projected[2]  # 归一化
            
            # 获取观测到的2D点坐标
            try:
                features = self.db.load_keypoints(image_id, load_desc=False)
                kp_idx = point.observations[image_id]
                observed = features.keypoints_xy[kp_idx]
                
                # 计算重投影误差
                error = np.linalg.norm(projected - observed)
                
                # 如果误差过大，标记为异常点
                if error > self.config.filter_max_reproj_error:
                    outliers.append(point_id)
            except Exception as e:
                self.logger.warning(f"计算点 {point_id} 的重投影误差失败: {str(e)}")
        
        # 移除异常点
        for point_id in outliers:
            # 从点的观测中移除当前图像
            if point_id in self.scene.points3d:
                point = self.scene.points3d[point_id]
                if image_id in point.observations:
                    del point.observations[image_id]
                
                # 如果观测数量不足，完全移除该点
                if len(point.observations) < self.config.filter_min_track_length:
                    self.scene.remove_point3d(point_id)
        
        self.logger.debug(f"从图像 {image_id} 中过滤了 {len(outliers)} 个异常点")