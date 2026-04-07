"""
几何验证模块

使用pymagsac++进行几何验证，自适应选择Essential或Homography模型
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
import cv2
import pymagsac

from config import FeatureConfig
from core.data_models import VerificationResult, GeometryModel, MatchData
from storage.database import Database


class PymagsacVerifier:
    """基于pymagsac++的几何验证器
    
    对特征匹配进行几何验证，自适应选择Essential或Homography模型
    """
    
    def __init__(self, db: Database, config: FeatureConfig):
        """初始化几何验证器
        
        Args:
            db: 数据库实例
            config: 特征提取配置
        """
        self.db = db
        self.config = config
        self.logger = logging.getLogger("PymagsacVerifier")
    

    def verify_image_pairs(self, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """验证多对图像的几何一致性
        
        Args:
            pairs: 图像对列表 [(id1, id2), ...]
            
        Returns:
            验证通过的图像对列表 [(id1, id2), ...]
        """
        self.logger.info(f"开始几何验证 {len(pairs)} 对图像")
        
        valid_pairs = []
        for i, (id1, id2) in enumerate(pairs):
            try:
                result = self._verify_image_pair(id1, id2)
                if result is not None:
                    # 保存验证结果
                    self.db.save_verification(result)
                    
                    if result.is_valid:
                        valid_pairs.append((id1, id2))
                
                # 打印进度
                if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                    self.logger.info(f"已处理 {i + 1}/{len(pairs)} 对图像")
            except Exception as e:
                self.logger.error(f"验证图像对 ({id1}, {id2}) 时出错: {str(e)}")
        
        self.logger.info(f"验证通过 {len(valid_pairs)}/{len(pairs)} 对图像")
        return valid_pairs
    

    def _verify_image_pair(self, id1: int, id2: int) -> Optional[VerificationResult]:
        """验证一对图像的几何一致性
        
        使用MAGSAC++进行几何验证，自适应选择Essential、Fundamental或Homography模型
        参考COLMAP的实现逻辑
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            
        Returns:
            验证结果，包含几何验证的详细信息
        """
        try:
            # 加载匹配数据
            match_data = self.db.load_matches(id1, id2)
            if match_data is None or len(match_data.matches) < 8:  # 至少需要8对点
                return self._create_invalid_result(id1, id2)
            
            # 加载特征点
            features1 = self.db.load_keypoints(id1)
            features2 = self.db.load_keypoints(id2)
            
            # 加载相机内参
            K1 = self.db.load_camera_intrinsics(id1)
            K2 = self.db.load_camera_intrinsics(id2)
            
            if K1 is None or K2 is None:
                self.logger.warning(f"无法加载图像 {id1} 或 {id2} 的相机内参，使用基础矩阵模式")
                has_calibration = False
            else:
                has_calibration = True
            
            # 获取匹配点对
            pts1 = []
            pts2 = []
            valid_match_indices = []  # 跟踪有效匹配的索引
        
            for i, (idx1, idx2) in enumerate(match_data.matches):
                if idx1 < features1.num_keypoints and idx2 < features2.num_keypoints:
                    pts1.append(features1.keypoints_xy[idx1])
                    pts2.append(features2.keypoints_xy[idx2])
                    valid_match_indices.append(i)
            
            if len(pts1) < 8:
                return self._create_invalid_result(id1, id2)
            
            # 转换为numpy数组
            pts1 = np.array(pts1, dtype=np.float32)
            pts2 = np.array(pts2, dtype=np.float32)
            valid_match_indices = np.array(valid_match_indices, dtype=np.int32)
            
            # 设置RANSAC参数
            ransac_threshold = self.config.ransac_threshold
            ransac_confidence = self.config.ransac_confidence
            ransac_max_iters = self.config.ransac_max_iters
            min_inliers = max(15, int(0.1 * len(pts1)))  # 至少15个内点或10%
            
            # 1. 估计Homography矩阵 (适用于所有情况)
            H_result = None
            try:
                H, H_mask = cv2.findHomography(
                    pts1, pts2, 
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=ransac_threshold,
                    confidence=ransac_confidence,
                    maxIters=ransac_max_iters
                )
                
                if H is not None and H_mask is not None:
                    H_inlier_mask = H_mask.ravel().astype(bool)
                    H_inlier_count = np.sum(H_inlier_mask)
                    H_inlier_ratio = H_inlier_count / len(pts1)
                    
                    H_result = {
                        "model": H,
                        "inlier_mask": H_inlier_mask,
                        "inlier_count": H_inlier_count,
                        "inlier_ratio": H_inlier_ratio
                    }
                    self.logger.debug(f"H矩阵估计: {H_inlier_count}/{len(pts1)} 内点 ({H_inlier_ratio:.2f})")
            except Exception as e:
                self.logger.warning(f"H矩阵估计失败: {str(e)}")
            
            # 2. 如果有内参，估计Essential矩阵
            E_result = None
            if has_calibration:
                try:
                    # 归一化点坐标
                    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
                    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)
                    
                    E, E_mask = cv2.findEssentialMat(
                        pts1_norm, pts2_norm,
                        method=cv2.USAC_MAGSAC,
                        threshold=ransac_threshold / ((K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) / 4.0),  # 归一化阈值
                        prob=ransac_confidence,
                        maxIters=ransac_max_iters
                    )
                    
                    if E is not None and E_mask is not None:
                        E_inlier_mask = E_mask.ravel().astype(bool)
                        E_inlier_count = np.sum(E_inlier_mask)
                        E_inlier_ratio = E_inlier_count / len(pts1)
                        
                        # 从E矩阵恢复相对位姿
                        R, t = None, None
                        if E_inlier_count >= 5:  # 至少需要5个内点才能恢复位姿
                            try:
                                _, R, t, _ = cv2.recoverPose(E, pts1_norm[E_inlier_mask], pts2_norm[E_inlier_mask])
                                t = t.flatten()
                            except Exception as e:
                                self.logger.warning(f"从E矩阵恢复位姿失败: {str(e)}")
                        
                        E_result = {
                            "model": E,
                            "inlier_mask": E_inlier_mask,
                            "inlier_count": E_inlier_count,
                            "inlier_ratio": E_inlier_ratio,
                            "R": R,
                            "t": t
                        }
                        self.logger.debug(f"E矩阵估计: {E_inlier_count}/{len(pts1)} 内点 ({E_inlier_ratio:.2f})")
                except Exception as e:
                    self.logger.warning(f"E矩阵估计失败: {str(e)}")
            
            # 3. 估计Fundamental矩阵 (无需内参)
            F_result = None
            try:
                F, F_mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=ransac_threshold,
                    confidence=ransac_confidence,
                    maxIters=ransac_max_iters
                )
                
                if F is not None and F_mask is not None:
                    F_inlier_mask = F_mask.ravel().astype(bool)
                    F_inlier_count = np.sum(F_inlier_mask)
                    F_inlier_ratio = F_inlier_count / len(pts1)
                    
                    F_result = {
                        "model": F,
                        "inlier_mask": F_inlier_mask,
                        "inlier_count": F_inlier_count,
                        "inlier_ratio": F_inlier_ratio
                    }
                    self.logger.debug(f"F矩阵估计: {F_inlier_count}/{len(pts1)} 内点 ({F_inlier_ratio:.2f})")
            except Exception as e:
                self.logger.warning(f"F矩阵估计失败: {str(e)}")
            
            # 4. 检查所有模型是否都失败
            if (H_result is None or H_result["inlier_count"] < min_inliers) and \
                (E_result is None or E_result["inlier_count"] < min_inliers) and \
                (F_result is None or F_result["inlier_count"] < min_inliers):
                self.logger.debug(f"图像对 ({id1}, {id2}) 的所有几何模型验证失败")
                return self._create_invalid_result(id1, id2)
            
            # 5. 确定最佳模型 - 参考COLMAP的逻辑
            best_model = None
            model_type = GeometryModel.UNKNOWN
            R, t = None, None
            
            # 设置阈值参数
            min_E_F_ratio = 0.95  # E比F内点数比例阈值
            max_H_ratio = 0.8     # H比E/F内点数比例阈值，超过认为是平面场景
            
            # 计算模型间的内点比例
            E_F_ratio = 0 if E_result is None or F_result is None else E_result["inlier_count"] / F_result["inlier_count"]
            H_F_ratio = 0 if H_result is None or F_result is None else H_result["inlier_count"] / F_result["inlier_count"]
            H_E_ratio = 0 if H_result is None or E_result is None else H_result["inlier_count"] / E_result["inlier_count"]
            
            # 根据COLMAP逻辑选择模型
            if has_calibration and E_result is not None and E_result["inlier_count"] >= min_inliers and E_F_ratio > min_E_F_ratio:
                # 有内参，优先使用E矩阵
                if H_E_ratio > max_H_ratio:
                    # 平面场景
                    model_type = GeometryModel.HOMOGRAPHY
                    best_model = H_result if H_result["inlier_count"] > E_result["inlier_count"] else E_result
                    self.logger.debug(f"选择平面场景模型 (H/E比例: {H_E_ratio:.2f})")
                else:
                    # 一般场景，使用E矩阵
                    model_type = GeometryModel.ESSENTIAL
                    best_model = E_result
                    R, t = E_result["R"], E_result["t"]
                    self.logger.debug(f"选择Essential矩阵模型")
            elif F_result is not None and F_result["inlier_count"] >= min_inliers:
                # 无内参或E矩阵不可靠，使用F矩阵
                if H_F_ratio > max_H_ratio:
                    # 平面场景
                    model_type = GeometryModel.HOMOGRAPHY
                    best_model = H_result if H_result["inlier_count"] > F_result["inlier_count"] else F_result
                    self.logger.debug(f"选择平面场景模型 (H/F比例: {H_F_ratio:.2f})")
                else:
                    # 一般场景，使用F矩阵
                    model_type = GeometryModel.FUNDAMENTAL
                    best_model = F_result
                    self.logger.debug(f"选择Fundamental矩阵模型")
            elif H_result is not None and H_result["inlier_count"] >= min_inliers:
                # 只有H矩阵可用
                model_type = GeometryModel.HOMOGRAPHY
                best_model = H_result
                self.logger.debug(f"只有Homography矩阵可用，选择H矩阵模型")
            else:
                # 所有模型都不可靠
                return self._create_invalid_result(id1, id2)
            
            # 6. 如果是H矩阵且有内参，尝试分解H矩阵获取R和t
            if has_calibration and R is None and t is None:
                if model_type == GeometryModel.HOMOGRAPHY:
                    R, t = self._decompose_homography_matrix(H, K1, K2, pts1, pts2, best_model["inlier_mask"])
                elif model_type == GeometryModel.FUNDAMENTAL:
                    R, t = self._decompose_fundamental_matrix(F, K1, K2, pts1, pts2, best_model["inlier_mask"])
            
            # 7. 计算视差分数
            parallax_score = self._compute_parallax_score(pts1, pts2, best_model["inlier_mask"], K1, K2, R)
            
            # 8. 创建验证结果
            return self._create_verification_result(
                id1, id2, best_model["inlier_mask"], model_type, 
                E=E_result["model"] if E_result is not None else None,
                F=F_result["model"] if F_result is not None else None,
                H=H_result["model"] if H_result is not None else None,
                R=R, t=t,
                inlier_count=int(best_model["inlier_count"]),
                inlier_ratio=best_model["inlier_ratio"],
                parallax_score=parallax_score
            )
            
        except Exception as e:
            self.logger.error(f"验证图像对 ({id1}, {id2}) 时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._create_invalid_result(id1, id2)
    

    def _create_verification_result(self, id1: int, id2: int, inlier_mask: np.ndarray, model_type: GeometryModel,
                                        E: Optional[np.ndarray] = None,
                                        F: Optional[np.ndarray] = None,
                                        H: Optional[np.ndarray] = None,
                                        R: Optional[np.ndarray] = None,
                                        t: Optional[np.ndarray] = None,
                                        inlier_count: Optional[int] = None,
                                        inlier_ratio: Optional[float] = None,
                                        parallax_score: Optional[float] = None) -> VerificationResult:
        """创建验证结果并更新匹配
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            pts1: 第一个图像中的点坐标
            pts2: 第二个图像中的点坐标
            match_data: 原始匹配数据
            inlier_mask: 内点掩码
            valid_match_indices: 有效匹配索引
            model_type: 几何模型类型
            E: 可选，Essential矩阵 [3, 3]
            F: 可选，Fundamental矩阵 [3, 3]
            H: 可选，Homography矩阵 [3, 3]
            R: 可选，旋转矩阵 [3, 3]
            t: 可选，平移向量 [3]
            inlier_count: 可选，内点数量
            inlier_ratio: 可选，内点比例
            parallax_score: 可选，视差分数
            
        Returns:
            验证结果
        """
        
        # 创建验证结果，包含几何矩阵和位姿信息
        result = VerificationResult(
            img_id1=id1,
            img_id2=id2,
            is_valid=True,
            model_type=model_type,
            E_mat=E,
            F_mat=F,
            H_mat=H,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            inlier_ratio=float(inlier_ratio),
            relative_R=R,
            relative_t=t,
            parallax_score=float(parallax_score)
        )
        
        return result
    

    def _estimate_essential(self, points1: np.ndarray, points2: np.ndarray, 
                          K1: np.ndarray, K2: np.ndarray) -> Dict:
        """使用pymagsac++估计Essential矩阵
        
        Args:
            points1: 第一个图像中的点坐标 [N, 2]
            points2: 第二个图像中的点坐标 [N, 2]
            K1: 第一个相机的内参矩阵 [3, 3]
            K2: 第二个相机的内参矩阵 [3, 3]
            
        Returns:
            包含估计结果的字典
        """
        # 归一化点坐标
        points1_normalized = cv2.undistortPoints(points1, K1, None)
        points2_normalized = cv2.undistortPoints(points2, K2, None)
        
        # 重塑为Nx2数组
        points1_normalized = points1_normalized.reshape(-1, 2)
        points2_normalized = points2_normalized.reshape(-1, 2)
        
        # 使用pymagsac++估计Essential矩阵
        try:
            E, mask = pymagsac.findEssentialMatrix(
                points1_normalized, points2_normalized,
                threshold=self.config.magsac_threshold / max(K1[0, 0], K2[0, 0]),  # 像素阈值转换为归一化坐标阈值
                confidence=self.config.magsac_confidence,
                max_iters=self.config.magsac_max_iters
            )
            
            # 计算内点数量和比例
            inlier_mask = mask.astype(bool)
            inlier_count = np.sum(inlier_mask)
            inlier_ratio = inlier_count / len(points1) if len(points1) > 0 else 0
            
            # 从Essential矩阵恢复相对位姿
            R, t = self._decompose_essential_matrix(E, points1_normalized[inlier_mask], 
                                                  points2_normalized[inlier_mask])
            
            return {
                "model": E,
                "inlier_mask": inlier_mask,
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
                "R": R,
                "t": t
            }
        except Exception as e:
            self.logger.warning(f"Essential矩阵估计失败: {str(e)}")
            # 返回空结果
            dummy_mask = np.zeros(len(points1), dtype=bool)
            return {
                "model": np.eye(3),
                "inlier_mask": dummy_mask,
                "inlier_count": 0,
                "inlier_ratio": 0,
                "R": np.eye(3),
                "t": np.zeros(3)
            }
    
    def _estimate_homography(self, points1: np.ndarray, points2: np.ndarray) -> Dict:
        """使用pymagsac++估计Homography矩阵
        
        Args:
            points1: 第一个图像中的点坐标 [N, 2]
            points2: 第二个图像中的点坐标 [N, 2]
            
        Returns:
            包含估计结果的字典
        """
        # 使用pymagsac++估计Homography矩阵
        try:
            H, mask = pymagsac.findHomography(
                points1, points2,
                threshold=self.config.magsac_threshold,
                confidence=self.config.magsac_confidence,
                max_iters=self.config.magsac_max_iters
            )
            
            # 计算内点数量和比例
            inlier_mask = mask.astype(bool)
            inlier_count = np.sum(inlier_mask)
            inlier_ratio = inlier_count / len(points1) if len(points1) > 0 else 0
            
            return {
                "model": H,
                "inlier_mask": inlier_mask,
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio
            }
        except Exception as e:
            self.logger.warning(f"Homography矩阵估计失败: {str(e)}")
            # 返回空结果
            dummy_mask = np.zeros(len(points1), dtype=bool)
            return {
                "model": np.eye(3),
                "inlier_mask": dummy_mask,
                "inlier_count": 0,
                "inlier_ratio": 0
            }
    
    def _decompose_homography_matrix(self, H: np.ndarray, K1: np.ndarray, K2: np.ndarray,
                               points1: np.ndarray, points2: np.ndarray,
                               inlier_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """分解单应矩阵，恢复相对位姿
        
        使用OpenCV的decomposeHomographyMat分解H矩阵，并使用正深度约束选择最佳解
        
        Args:
            H: 单应矩阵 [3, 3]
            K1: 第一个相机的内参矩阵 [3, 3]
            K2: 第二个相机的内参矩阵 [3, 3]
            points1: 第一个图像中的点坐标 [N, 2]
            points2: 第二个图像中的点坐标 [N, 2]
            inlier_mask: 内点掩码 [N]
            
        Returns:
            (R, t) 元组，R为旋转矩阵 [3, 3]，t为平移向量 [3]
        """
        try:
            # 使用OpenCV分解H矩阵，得到多组可能的解
            num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K1)
            
            if num == 0:
                self.logger.warning(f"H矩阵分解失败，无有效解")
                return np.eye(3), np.zeros(3)
            
            self.logger.debug(f"从H矩阵分解出 {num} 组可能的R和t")
            
            # 只使用内点进行验证
            inlier_pts1 = points1[inlier_mask]
            inlier_pts2 = points2[inlier_mask]
            
            # 归一化点坐标
            pts1_norm = cv2.undistortPoints(inlier_pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
            pts2_norm = cv2.undistortPoints(inlier_pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)
            
            # 使用正深度约束选择最佳解
            max_positive = 0
            best_R = Rs[0]
            best_t = ts[0].flatten()
            
            for i, (R, t) in enumerate(zip(Rs, ts)):
                # 确保t是一维向量
                t_flat = t.flatten()
                
                # 构建第一个相机的投影矩阵（假设为单位矩阵）
                P1 = np.eye(3, 4)
                
                # 构建第二个相机的投影矩阵
                P2 = np.column_stack((R, t_flat.reshape(3, 1)))
                
                # 三角化点
                points_4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
                
                # 转换为非齐次坐标
                points_3d = points_4d[:3] / points_4d[3]
                
                # 计算在两个相机前方的点数量
                # 第一个相机：Z > 0
                # 第二个相机：R*X + t 的Z分量 > 0
                points_in_cam2 = R @ points_3d + t_flat.reshape(3, 1)
                
                positive_z1 = np.sum(points_3d[2] > 0)
                positive_z2 = np.sum(points_in_cam2[2] > 0)
                
                total_positive = positive_z1 + positive_z2
                
                self.logger.debug(f"位姿解{i+1}: 相机1前方点数={positive_z1}, 相机2前方点数={positive_z2}, 总数={total_positive}")
                
                if total_positive > max_positive:
                    max_positive = total_positive
                    best_R = R
                    best_t = t_flat
            
            # 归一化t向量
            if np.linalg.norm(best_t) > 0:
                best_t = best_t / np.linalg.norm(best_t)
            
            self.logger.debug(f"从{num}组位姿解中选择最佳解，共有{max_positive}/{2*len(pts1_norm)}个点满足正深度约束")
            return best_R, best_t
        
        except Exception as e:
            self.logger.warning(f"H矩阵分解失败: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return np.eye(3), np.zeros(3)
    

    def _decompose_fundamental_matrix(self, F: np.ndarray, K1: np.ndarray, K2: np.ndarray,
                           points1: np.ndarray, points2: np.ndarray,
                           inlier_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从基础矩阵恢复相对位姿
        
        使用相机内参将F矩阵转换为E矩阵，然后分解E矩阵得到R和t
        
        Args:
            F: 基础矩阵 [3, 3]
            K1: 第一个相机的内参矩阵 [3, 3]
            K2: 第二个相机的内参矩阵 [3, 3]
            points1: 第一个图像中的点坐标 [N, 2]
            points2: 第二个图像中的点坐标 [N, 2]
            inlier_mask: 内点掩码 [N]
            
        Returns:
            (R, t) 元组，R为旋转矩阵 [3, 3]，t为平移向量 [3]
        """
        try:
            # 从F矩阵恢复E矩阵: E = K2.T * F * K1
            E_from_F = K2.T @ F @ K1
            
            # 使用内点掩码筛选匹配点
            inlier_pts1 = points1[inlier_mask]
            inlier_pts2 = points2[inlier_mask]
            
            # 归一化点坐标
            pts1_norm = cv2.undistortPoints(inlier_pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
            pts2_norm = cv2.undistortPoints(inlier_pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)
            
            # 使用recoverPose从E矩阵恢复R和t
            _, R, t, _ = cv2.recoverPose(E_from_F, pts1_norm, pts2_norm)
            t = t.flatten()
            
            # 归一化t向量
            if np.linalg.norm(t) > 0:
                t = t / np.linalg.norm(t)
                
            self.logger.debug(f"从F矩阵通过内参恢复出R和t")
            return R, t
            
        except Exception as e:
            self.logger.warning(f"F矩阵分解失败: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return np.eye(3), np.zeros(3)


    def _compute_parallax_score(self,
                             points1: np.ndarray,   
                             points2: np.ndarray,
                             inlier_mask: np.ndarray,
                             K1: np.ndarray = None,  
                             K2: np.ndarray = None,
                             R: np.ndarray = None,   # 相对旋转，有则去除旋转分量
                             ) -> float:
        """计算视差分数（中位数三角化角）
        
        Returns:
            视差角中位数（角度，°），越大表示基线越好
            注意：返回角度值而非[0,1]，由调用方决定如何归一化
        """
        if np.sum(inlier_mask) < 10:
            return 0.0

        pts1 = points1[inlier_mask]
        pts2 = points2[inlier_mask]

        # Step1: 转换到归一化相机坐标（消除焦距影响）
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])  # [N,3]
        pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
        rays1 = (np.linalg.inv(K1) @ pts1_h.T).T  # [N,3]
        rays2 = (np.linalg.inv(K2) @ pts2_h.T).T


        # Step2: 归一化射线方向
        rays1 = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
        rays2 = rays2 / np.linalg.norm(rays2, axis=1, keepdims=True)

        # Step3: 去除旋转分量（可选）
        rays2_unrotated = (R.T @ rays2.T).T

        # Step4: 计算射线夹角
        cos_angles = np.clip(
            np.sum(rays1 * rays2_unrotated, axis=1),
            -1.0, 1.0
        )
        angles_deg = np.degrees(np.arccos(cos_angles))

        # Step5: 取中位数（与OpenMVG一致）
        median_angle = float(np.median(angles_deg))

        return median_angle

    

    def _create_invalid_result(self, id1: int, id2: int) -> VerificationResult:
        """创建无效的验证结果
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            
        Returns:
            无效的验证结果
        """
        return VerificationResult(
            img_id1=id1,
            img_id2=id2,
            is_valid=False,
            model_type=GeometryModel.UNKNOWN,
            inlier_count=0,
            inlier_ratio=0.0,
            parallax_score=0.0
        )