"""
SFM初始化模块

选择合适的初始图像对，三角化初始点云
"""
import cv2
import math
import logging
import numpy as np
from typing import List, Tuple, Optional

from config import SFMConfig
from storage.database import Database
from core.scene import SceneManager, View
from ba.local_ba import LocalBundleAdjuster
from core.geo_utils import GPSLocalConverter



class SFMInitializer:
    """SFM初始化器
    
    负责选择初始图像对，分解Essential矩阵，三角化初始点云
    """
    
    def __init__(self, db: Database, config: SFMConfig):
        """初始化SFM初始化器
        
        Args:
            db: 数据库实例
            config: SFM配置
        """
        self.db = db
        self.config = config
        self.logger = logging.getLogger("SFMInitializer")
    

    def initialize(self, scene: SceneManager, ba_optimizer: LocalBundleAdjuster) -> Tuple[bool, List[int]]:
        """初始化SFM重建
        
        选择合适的初始图像对，分解Essential矩阵，三角化初始点云
        
        Args:
            scene: 场景管理器
            
        Returns:
            (成功标志, 初始化使用的图像ID列表)
        """
        self.logger.info("开始SFM初始化")
        
        # 1. 选择最佳初始对
        id1, id2 = self._select_initial_pair()
        if id1 is None or id2 is None:
            self.logger.error("无法找到合适的初始图像对")
            return False, []
        
        self.logger.info(f"选择初始图像对: ({id1}, {id2})")
        
        # 2. 加载图像元数据和几何验证结果
        try:
            meta1 = self.db.load_image_meta(id1)
            meta2 = self.db.load_image_meta(id2)
            verification = self.db.load_verification(id1, id2)
        except Exception as e:
            self.logger.error(f"加载初始图像对数据失败: {str(e)}")
            return False, []
        
        # 2. 设置第一个视图为世界坐标系原点
        view1 = View(
            image_id=id1,
            K=meta1.K,
            R=np.eye(3),  # 单位旋转
            t=np.zeros((3, 1)),  # 零平移
            gps_local_xyz=np.zeros(3),
            is_registered=True
        )
        
        # 3. 根据Essential矩阵分解设置第二个视图
        R = verification.relative_R
        t = verification.relative_t.reshape(3, 1)

        # 建立gps转换器，把gps坐标转换为ENU坐标
        GPSLocalConverter.initialize(meta1.gps_lat, meta1.gps_lon, meta1.gps_alt)
        local_xyz = GPSLocalConverter.to_local_enu(meta2.gps_lat, meta2.gps_lon, meta2.gps_alt)

        view2 = View(
            image_id=id2,
            K=meta2.K,
            R=R,
            t=t,
            gps_local_xyz=local_xyz,
            is_registered=True
        )
        
        # 4. 将相机添加到场景
        scene.add_view(view1)
        scene.add_view(view2)
        
        # 5. 三角化初始点云
        success = self._triangulate_initial_points(scene, id1, id2)
        
        if not success:
            self.logger.error("三角化初始点云失败")
            return False, []
        
        # 6. 对初始重建执行BA优化
        try:
            # 执行初始BA优化
            self.logger.info("执行初始对BA优化...")
            ba_result = ba_optimizer.optimize(
                scene=scene,
                central_image_id=id2,  # 以第二个相机为中心
                fix_cameras=[id1],     # 固定第一个相机（参考坐标系）
                max_iterations=50      # 初始BA可以多迭代几次
            )
            
        except Exception as e:
            self.logger.error(f"初始BA失败: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            # BA失败不应该导致整个初始化失败，继续执行


        self.logger.info("SFM初始化成功")
        return True, [id1, id2]
    

    def _select_initial_pair(self) -> Tuple[Optional[int], Optional[int]]:
        """选择最佳初始图像对
        
        选择具有足够内点和良好视差的图像对。
        评分策略：综合视差角（三段式）和内点数量，优先选视差适中且内点多的对。
        
        Returns:
            (id1, id2) 元组，如果没有找到合适的对则返回(None, None)
        """
        # ── 1. 加载所有有效的几何验证对 ──────────────────────────────────────
        try:
            valid_pairs = self.db.load_valid_pairs()
        except Exception as e:
            self.logger.error(f"加载有效几何验证对失败: {str(e)}")
            return None, None

        if not valid_pairs:
            self.logger.error("没有有效的几何验证对")
            return None, None

        self.logger.debug(f"共加载 {len(valid_pairs)} 个有效几何验证对")

        # ── 2. 读取评分相关配置 ───────────────────────────────────────────────
        init_required_min_angle = self.config.init_required_min_angle    # 最低视差角门槛，如 3.0°
        init_optimal_angle      = self.config.init_optimal_angle         # 最优视差角，如 15.0°
        init_limit_max_angle    = self.config.init_limit_max_angle       # 视差角上限，如 33.0°
        init_inlier_norm_ref    = self.config.init_inlier_norm_ref       # 内点数归一化参考，如 300
        w_parallax              = getattr(self.config, 'init_weight_parallax', 0.6)
        w_inlier                = getattr(self.config, 'init_weight_inlier',   0.4)

        # ── 3. 遍历筛选 ───────────────────────────────────────────────────────
        candidates  = []
        skip_reasons = {"load_fail": 0, "invalid": 0}

        for id1, id2 in valid_pairs:
            # 3.1 加载验证结果
            try:
                verification = self.db.load_verification(id1, id2)
            except Exception as e:
                self.logger.warning(f"加载图像对 ({id1}, {id2}) 的验证结果失败: {str(e)}")
                skip_reasons["load_fail"] += 1
                continue

            # 3.2 is_valid 已经包含了内点数/内点比/视差角等基础过滤
            if not verification.is_valid:
                skip_reasons["invalid"] += 1
                continue

            parallax    = verification.parallax_score
            inlier_count = verification.inlier_count
            inlier_ratio = verification.inlier_ratio

            # 3.3 视差角评分（三段式）
            #
            #  score
            #  1.0 |              *----------*
            #  0.5 |         *               \
            #      |      *                   \
            #  0.0 |*____________________________
            #      0  min°      optimal°      max°
            #
            if parallax <= init_optimal_angle:
                parallax_score = 0.5 + 0.5 * ((parallax - init_required_min_angle) / (init_optimal_angle - init_required_min_angle))
            else:
                parallax_score = 1.0 - 0.3 * min((parallax - init_optimal_angle) / (init_limit_max_angle - init_optimal_angle), 1.0)

            # 3.4 内点数评分（对数归一化，上限截断到 1.0）
            inlier_score = min(math.log1p(inlier_count) / math.log1p(init_inlier_norm_ref), 1.0)

            # 3.5 综合评分
            final_score = w_parallax * parallax_score + w_inlier * inlier_score

            candidates.append({
                "id1":          id1,
                "id2":          id2,
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
                "parallax":     parallax,
                "parallax_score": parallax_score,
                "inlier_score":   inlier_score,
                "score":          final_score,
            })

        # ── 4. 过滤统计日志 ───────────────────────────────────────────────────
        self.logger.debug(
            f"候选过滤统计 → "
            f"加载失败:{skip_reasons['load_fail']}  "
            f"is_valid不通过:{skip_reasons['invalid']}  "
            f"通过:{len(candidates)}"
        )

        if not candidates:
            self.logger.error(
                f"没有满足条件的初始图像对候选 "
                f"(required_min_angle={init_required_min_angle:.1f}°, "
                f"optimal_angle={init_optimal_angle:.1f}°, "
                f"limit_max_angle={init_limit_max_angle:.1f}°)"
            )
            return None, None

        # ── 5. 排序并输出 Top-5 ───────────────────────────────────────────────
        candidates.sort(key=lambda x: x["score"], reverse=True)

        top_n = min(5, len(candidates))
        self.logger.debug(f"Top-{top_n} 初始对候选：")
        for i, c in enumerate(candidates[:top_n]):
            self.logger.debug(
                f"  [{i+1}] ({c['id1']:>4}, {c['id2']:>4})  "
                f"score={c['score']:.4f}  "
                f"parallax={c['parallax']:>6.2f}°(s={c['parallax_score']:.3f})  "
                f"inliers={c['inlier_count']:>4}(s={c['inlier_score']:.3f})  "
                f"ratio={c['inlier_ratio']:.3f}"
            )

        best = candidates[0]
        self.logger.info(
            f"选择初始图像对: ({best['id1']}, {best['id2']})  "
            f"score={best['score']:.4f}  "
            f"parallax={best['parallax']:.2f}°  "
            f"inliers={best['inlier_count']}"
        )
        return best["id1"], best["id2"]
    

    def _triangulate_initial_points(self, scene: SceneManager, id1: int, id2: int) -> bool:
        """三角化初始点云
        
        Args:
            scene: 场景管理器
            id1: 第一个图像ID
            id2: 第二个图像ID
            inlier_mask: 内点掩码
            
        Returns:
            是否成功三角化
        """
        # 加载特征点和匹配
        try:
            features1 = self.db.load_keypoints(id1, load_desc=False)
            features2 = self.db.load_keypoints(id2, load_desc=False)
            matches = self.db.load_inlier_matches(id1, id2)
        except Exception as e:
            self.logger.error(f"加载特征点和匹配失败: {str(e)}")
            return False
        
        # 获取相机
        view1 = scene.views[id1]
        view2 = scene.views[id2]
        
        # 构建投影矩阵
        P1 = view1.K @ np.hstack((view1.R, view1.t))
        P2 = view2.K @ np.hstack((view2.R, view2.t))
        
        # 提取匹配点坐标
        points1 = features1.keypoints_xy[matches.matches[:, 0]]
        points2 = features2.keypoints_xy[matches.matches[:, 1]]
        
        # 三角化点
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        
        # 转换为齐次坐标
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T  # 转置为 Nx3
        
        # 检查点是否在相机前方
        valid_points = self._check_point_in_front(points_3d, view1, view2)
        
        # 检查三角化角度
        valid_angles = self._check_triangulation_angle(points1, points2, view1, view2)
        
        # 综合两个条件
        valid_mask = np.logical_and(valid_points, valid_angles)
        
        # 添加有效的3D点到场景
        added_count = 0
        for i, valid in enumerate(valid_mask):
            if valid:
                kp_idx1 = matches.matches[i, 0]
                kp_idx2 = matches.matches[i, 1]
                
                # 创建观测字典
                observations = {id1: kp_idx1, id2: kp_idx2}
                
                # 添加3D点
                scene.add_point3d(points_3d[i], observations)
                added_count += 1
        
        self.logger.info(f"三角化了 {added_count} 个初始3D点")
        return added_count > 0
    

    def _check_point_in_front(self, points_3d: np.ndarray, view1: View, view2: View) -> np.ndarray:
        """检查点是否在相机前方
        
        Args:
            points_3d: 3D点坐标 [N, 3]
            view1: 第一个视图
            view2: 第二个视图
            
        Returns:
            布尔掩码 [N]
        """
        # X_cam = R @ X_world + t
        # points_3d: (N,3) → 转置 (3,N)
        
        # 相机1坐标系下的Z值
        # view1.R=(3,3), view1.t=(3,1), points_3d.T=(3,N)
        cam1_z = (view1.R @ points_3d.T + view1.t)[2]   # (N,)
        
        # 相机2坐标系下的Z值
        cam2_z = (view2.R @ points_3d.T + view2.t)[2]   # (N,)
        
        return (cam1_z > 0) & (cam2_z > 0)

    
    
    def _check_triangulation_angle(self, points1: np.ndarray, points2: np.ndarray, 
                                 view1: View, view2: View) -> np.ndarray:
        N = len(points1)
        
        # 构建齐次坐标 (N,3)
        ones = np.ones((N, 1))
        p1_h = np.hstack([points1, ones])   # (N,3)
        p2_h = np.hstack([points2, ones])   # (N,3)
        
        # 批量反投影：K_inv @ p^T → (3,N)，再转置为 (N,3)
        K1_inv = np.linalg.inv(view1.K)
        K2_inv = np.linalg.inv(view2.K)
        
        rays1 = (K1_inv @ p1_h.T).T         # (N,3)
        rays2 = (K2_inv @ p2_h.T).T         # (N,3)
        
        # 旋转到世界坐标系：R.T @ ray^T → (3,N)，再转置
        rays1 = (view1.R.T @ rays1.T).T     # (N,3)
        rays2 = (view2.R.T @ rays2.T).T     # (N,3)
        
        # 批量归一化
        rays1 = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
        rays2 = rays2 / np.linalg.norm(rays2, axis=1, keepdims=True)
        
        # 计算夹角
        cos_angles = np.clip(np.sum(rays1 * rays2, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angles))
        
        min_angle = self.config.min_triangulation_angle_deg
        max_angle = self.config.max_triangulation_angle_deg
        
        return (angles >= min_angle) & (angles <= max_angle)
