"""
PLY点云导出模块

负责将重建结果导出为PLY格式的点云文件
"""
import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import struct
from pathlib import Path

from core.scene import SceneManager, View, Point3D


class PLYWriter:
    """PLY点云写入器
    
    将重建结果导出为PLY格式的点云文件
    """
    
    def __init__(self, output_path: str):
        """初始化PLY点云写入器
        
        Args:
            output_path: 输出文件路径
        """
        self.output_path = output_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def write_points(self, scene: SceneManager, with_cameras: bool = True, with_normals: bool = False) -> None:
        """将点云写入PLY文件
        
        Args:
            scene: 场景管理器
            with_cameras: 是否包含相机位置
            with_normals: 是否计算并包含法向量
        """
        self.logger.info(f"将点云写入PLY文件: {self.output_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 收集有效的3D点
        valid_points = []
        for point_id, point in scene.points3d.items():
            if not point.is_outlier and len(point.observations) >= 2:
                valid_points.append(point)
        
        # 收集有效的相机
        valid_cameras = []
        if with_cameras:
            for image_id, camera in scene.cameras.items():
                if camera.is_registered:
                    valid_cameras.append(camera)
        
        # 计算法向量（如果需要）
        normals = None
        if with_normals:
            normals = self._compute_normals(scene, valid_points)
        
        # 写入PLY文件
        with open(self.output_path, 'wb') as f:
            # 写入PLY头
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {len(valid_points) + len(valid_cameras)}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            
            if with_normals:
                f.write(b"property float nx\n")
                f.write(b"property float ny\n")
                f.write(b"property float nz\n")
            
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
            f.write(b"end_header\n")
            
            # 写入3D点
            for i, point in enumerate(valid_points):
                # 点坐标
                f.write(struct.pack("<fff", point.xyz[0], point.xyz[1], point.xyz[2]))
                
                # 法向量
                if with_normals:
                    f.write(struct.pack("<fff", normals[i][0], normals[i][1], normals[i][2]))
                
                # 颜色（默认白色）
                f.write(struct.pack("<BBB", 255, 255, 255))
            
            # 写入相机位置
            for camera in valid_cameras:
                # 计算相机中心（世界坐标系）
                C = -camera.R.T @ camera.t
                
                # 相机位置
                f.write(struct.pack("<fff", C[0], C[1], C[2]))
                
                # 法向量（如果需要）
                if with_normals:
                    f.write(struct.pack("<fff", 0, 0, 0))
                
                # 颜色（红色表示相机）
                f.write(struct.pack("<BBB", 255, 0, 0))
        
        self.logger.info(f"PLY点云写入完成，包含 {len(valid_points)} 个点和 {len(valid_cameras)} 个相机")
    
    def _compute_normals(self, scene: SceneManager, points: List[Point3D]) -> np.ndarray:
        """计算点云法向量
        
        使用PCA方法计算每个点的法向量
        
        Args:
            scene: 场景管理器
            points: 3D点列表
            
        Returns:
            法向量数组 (N, 3)
        """
        normals = np.zeros((len(points), 3))
        
        for i, point in enumerate(points):
            # 收集观测该点的相机中心
            camera_centers = []
            for image_id in point.observations:
                if image_id in scene.cameras:
                    camera = scene.cameras[image_id]
                    if camera.is_registered:
                        # 计算相机中心（世界坐标系）
                        C = -camera.R.T @ camera.t
                        camera_centers.append(C.flatten())
            
            if len(camera_centers) < 3:
                # 如果观测不足，使用默认法向量
                normals[i] = np.array([0, 0, 1])
                continue
            
            # 将相机中心转换为相对于点的向量
            vectors = np.array(camera_centers) - point.xyz
            
            # 归一化
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            
            # 使用PCA计算主方向
            try:
                cov = vectors.T @ vectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # 最小特征值对应的特征向量即为法向量
                normal = eigenvectors[:, 0]
                
                # 确保法向量指向外部（朝向相机）
                if np.mean(vectors @ normal) < 0:
                    normal = -normal
                
                normals[i] = normal
            except np.linalg.LinAlgError:
                # 如果PCA失败，使用默认法向量
                normals[i] = np.array([0, 0, 1])
        
        return normals


class PLYReader:
    """PLY点云读取器
    
    从PLY文件读取点云数据
    """
    
    def __init__(self, input_path: str):
        """初始化PLY点云读取器
        
        Args:
            input_path: 输入文件路径
        """
        self.input_path = input_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def read_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """从PLY文件读取点云
        
        Returns:
            (点坐标数组 (N, 3), 点颜色数组 (N, 3))
        """
        self.logger.info(f"从PLY文件读取点云: {self.input_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"找不到PLY文件: {self.input_path}")
        
        # 读取PLY文件
        with open(self.input_path, 'rb') as f:
            # 读取PLY头
            header_lines = []
            line = f.readline().strip()
            while line != b"end_header":
                header_lines.append(line.decode('utf-8'))
                line = f.readline().strip()
            
            # 解析头信息
            vertex_count = 0
            has_normals = False
            for line in header_lines:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("property float nx"):
                    has_normals = True
            
            # 读取点数据
            points = np.zeros((vertex_count, 3), dtype=np.float32)
            colors = np.zeros((vertex_count, 3), dtype=np.uint8)
            
            for i in range(vertex_count):
                # 读取点坐标
                x, y, z = struct.unpack("<fff", f.read(12))
                points[i] = [x, y, z]
                
                # 跳过法向量（如果有）
                if has_normals:
                    f.read(12)  # 跳过3个float
                
                # 读取颜色
                r, g, b = struct.unpack("<BBB", f.read(3))
                colors[i] = [r, g, b]
        
        self.logger.info(f"PLY点云读取完成，包含 {vertex_count} 个点")
        return points, colors