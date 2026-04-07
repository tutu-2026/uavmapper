"""
核心数据结构模块

定义项目中使用的主要数据结构和数据模型
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


@dataclass
class ImageMeta:
    """图像元数据类
    
    存储图像的基本信息、相机参数和GPS/姿态数据
    """
    image_id: int
    image_path: str
    width: int
    height: int
    focal_length_px: float
    cx: float  # 主点x坐标
    cy: float  # 主点y坐标
    k1: float = 0.0  # 径向畸变系数k1
    k2: float = 0.0  # 径向畸变系数k2
    p1: float = 0.0  # 切向畸变系数p1
    p2: float = 0.0  # 切向畸变系数p2
    k3: float = 0.0  # 径向畸变系数k3
    camera_model: str = "PINHOLE_CAMERA_BROWN"  # 相机模型
    camera_serial: str = ""  # 相机序列号
    gps_lat: Optional[float] = None  # GPS纬度
    gps_lon: Optional[float] = None  # GPS经度
    gps_alt: Optional[float] = None  # GPS高度
    yaw: Optional[float] = None  # 偏航角(度)
    pitch: Optional[float] = None  # 俯仰角(度)
    roll: Optional[float] = None  # 横滚角(度)
    capture_time: Optional[str] = None  # 拍摄时间
    
    @property
    def K(self) -> np.ndarray:
        """获取3x3内参矩阵
        
        Returns:
            3x3内参矩阵
        """
        return np.array([
            [self.focal_length_px, 0, self.cx],
            [0, self.focal_length_px, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    

    @property
    def has_gps(self) -> bool:
        """检查是否有GPS数据
        
        Returns:
            如果有完整的GPS数据则返回True
        """
        return (self.gps_lat is not None and 
                self.gps_lon is not None and 
                self.gps_alt is not None)
    
    
    @property
    def has_pose(self) -> bool:
        """检查是否有姿态数据
        
        Returns:
            如果有完整的姿态数据则返回True
        """
        return (self.yaw is not None and 
                self.pitch is not None and 
                self.roll is not None)

    @property
    def dist_coeffs(self) -> np.ndarray:
        """获取畸变系数数组
        
        Returns:
            畸变系数数组 [k1, k2, p1, p2, k3]
        """
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)


@dataclass
class FeatureData:
    """特征数据类
    
    存储图像的特征点和描述符
    """
    image_id: int
    num_keypoints: int
    keypoints_xy: np.ndarray    # (N, 2) float32，像素坐标
    scales: np.ndarray          # (N,)   float32
    orientations: np.ndarray    # (N,)   float32，弧度
    scores: np.ndarray          # (N,)   float32
    descriptors: np.ndarray     # (N, 128) float32，RootSIFT描述符


@dataclass
class MatchData:
    """匹配数据类
    
    存储两张图像之间的特征匹配
    """
    img_id1: int
    img_id2: int     # 存储时保证 id1 < id2
    matches: np.ndarray             # (M, 2) int32， 特征点索引对
    scores: np.ndarray              # (M,)   float32，匹配置信度


class GeometryModel(Enum):
    """几何模型类型枚举"""
    ESSENTIAL = "E"
    FUNDAMENTAL = "F"
    HOMOGRAPHY = "H"
    UNKNOWN = "?"


@dataclass
class VerificationResult:
    """几何验证结果类
    
    存储两视图几何验证的结果
    """
    img_id1: int
    img_id2: int
    is_valid: bool
    model_type: GeometryModel
    E_mat: Optional[np.ndarray] = None     # (3, 3) Essential Matrix
    F_mat: Optional[np.ndarray] = None     # (3, 3) Fundamental Matrix
    H_mat: Optional[np.ndarray] = None     # (3, 3) Homography Matrix
    inlier_mask: Optional[np.ndarray] = None  # (M,) bool，对应 MatchData.matches
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    relative_R: Optional[np.ndarray] = None  # (3, 3) 相对旋转（E矩阵分解）
    relative_t: Optional[np.ndarray] = None  # (3,)   相对平移（单位向量）
    parallax_score: float = 0.0             # 用于初始对选择