"""
COLMAP格式读写模块

负责读写COLMAP格式的相机、图像和3D点数据
"""
import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import shutil

from core.scene import SceneManager, View, Point3D
from core.data_models import ImageMeta


class COLMAPWriter:
    """COLMAP格式写入器
    
    将重建结果写入COLMAP格式的文本文件
    """
    
    def __init__(self, output_dir: str):
        """初始化COLMAP格式写入器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # COLMAP格式文件路径
        self.cameras_path = os.path.join(output_dir, "cameras.txt")
        self.images_path = os.path.join(output_dir, "images.txt")
        self.points3d_path = os.path.join(output_dir, "points3D.txt")
    
    def write_scene(self, scene: SceneManager, image_metas: Dict[int, ImageMeta]) -> None:
        """将场景写入COLMAP格式
        
        Args:
            scene: 场景管理器
            image_metas: 图像元数据字典 {image_id: ImageMeta}
        """
        self.logger.info(f"将场景写入COLMAP格式: {self.output_dir}")
        
        # 写入相机参数
        self._write_cameras(scene, image_metas)
        
        # 写入图像参数
        self._write_images(scene)
        
        # 写入3D点
        self._write_points3d(scene)
        
        self.logger.info(f"COLMAP格式写入完成")
    
    def _write_cameras(self, scene: SceneManager, image_metas: Dict[int, ImageMeta]) -> None:
        """写入相机参数
        
        Args:
            scene: 场景管理器
            image_metas: 图像元数据字典
        """
        with open(self.cameras_path, 'w') as f:
            # 写入文件头
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: {}\n".format(len(scene.cameras)))
            
            # 写入每个相机的参数
            for image_id, camera in scene.cameras.items():
                if not camera.is_registered:
                    continue
                
                if image_id not in image_metas:
                    self.logger.warning(f"找不到图像 {image_id} 的元数据")
                    continue
                
                meta = image_metas[image_id]
                
                # COLMAP相机模型
                # SIMPLE_RADIAL: f, cx, cy, k1
                # RADIAL: f, cx, cy, k1, k2
                model = "SIMPLE_RADIAL"
                params = [camera.K[0, 0], camera.K[0, 2], camera.K[1, 2], meta.k1]
                
                # 写入相机参数行
                f.write("{} {} {} {} {}\n".format(
                    image_id, model, meta.width, meta.height, " ".join(map(str, params))
                ))
    
    def _write_images(self, scene: SceneManager) -> None:
        """写入图像参数
        
        Args:
            scene: 场景管理器
        """
        with open(self.images_path, 'w') as f:
            # 写入文件头
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write("# Number of images: {}\n".format(len(scene.cameras)))
            
            # 写入每个图像的参数
            for image_id, camera in scene.cameras.items():
                if not camera.is_registered:
                    continue
                
                # 计算四元数
                R = camera.R.T  # COLMAP使用相机到世界的变换
                q = self._rotation_matrix_to_quaternion(R)
                
                # 计算相机中心（世界坐标系）
                C = -R @ camera.t
                
                # 获取图像名称
                image_name = os.path.basename(camera.image_path) if hasattr(camera, 'image_path') else f"image_{image_id}.jpg"
                
                # 写入图像参数行
                f.write("{} {} {} {} {} {} {} {} {} {}\n".format(
                    image_id, q[0], q[1], q[2], q[3], C[0], C[1], C[2], image_id, image_name
                ))
                
                # 写入2D点和对应的3D点ID
                points2d_line = []
                
                # 收集该图像观测到的所有3D点
                for point_id, point in scene.points3d.items():
                    if image_id in point.observations and not point.is_outlier:
                        keypoint_idx = point.observations[image_id]
                        
                        # 加载特征点坐标
                        try:
                            features = scene._db.load_keypoints(image_id, load_desc=False)
                            keypoint = features.keypoints_xy[keypoint_idx]
                            points2d_line.append(f"{keypoint[0]} {keypoint[1]} {point_id}")
                        except Exception as e:
                            self.logger.warning(f"加载图像 {image_id} 的特征点失败: {str(e)}")
                
                # 写入2D点行
                f.write(" ".join(points2d_line) + "\n")
    
    def _write_points3d(self, scene: SceneManager) -> None:
        """写入3D点
        
        Args:
            scene: 场景管理器
        """
        with open(self.points3d_path, 'w') as f:
            # 写入文件头
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: {}\n".format(len(scene.points3d)))
            
            # 写入每个3D点
            for point_id, point in scene.points3d.items():
                if point.is_outlier:
                    continue
                
                # 默认颜色（白色）
                color = [255, 255, 255]
                
                # 写入3D点行
                f.write("{} {} {} {} {} {} {} {} ".format(
                    point_id, point.xyz[0], point.xyz[1], point.xyz[2],
                    color[0], color[1], color[2], 0.0
                ))
                
                # 写入观测轨迹
                track = []
                for image_id, keypoint_idx in point.observations.items():
                    track.append(f"{image_id} {keypoint_idx}")
                
                f.write(" ".join(track) + "\n")
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """将旋转矩阵转换为四元数
        
        Args:
            R: 3x3旋转矩阵
            
        Returns:
            四元数 [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])


class COLMAPReader:
    """COLMAP格式读取器
    
    从COLMAP格式的文本文件读取重建结果
    """
    
    def __init__(self, input_dir: str):
        """初始化COLMAP格式读取器
        
        Args:
            input_dir: 输入目录路径
        """
        self.input_dir = input_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # COLMAP格式文件路径
        self.cameras_path = os.path.join(input_dir, "cameras.txt")
        self.images_path = os.path.join(input_dir, "images.txt")
        self.points3d_path = os.path.join(input_dir, "points3D.txt")
        
        # 检查文件是否存在
        if not os.path.exists(self.cameras_path):
            raise FileNotFoundError(f"找不到相机文件: {self.cameras_path}")
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(f"找不到图像文件: {self.images_path}")
        if not os.path.exists(self.points3d_path):
            raise FileNotFoundError(f"找不到3D点文件: {self.points3d_path}")
    
    def read_scene(self) -> SceneManager:
        """从COLMAP格式读取场景
        
        Returns:
            场景管理器
        """
        self.logger.info(f"从COLMAP格式读取场景: {self.input_dir}")
        
        # 创建场景管理器
        scene = SceneManager()
        
        # 读取相机参数
        cameras = self._read_cameras()
        
        # 读取图像参数
        images = self._read_images(cameras)
        
        # 读取3D点
        points3d = self._read_points3d()
        
        # 填充场景管理器
        for image_id, (K, R, t, image_path) in images.items():
            view = View(
                image_id=image_id,
                K=K,
                R=R,
                t=t,
                gps_local_xyz=None,  # COLMAP格式不包含GPS信息
                is_registered=True
            )
            if image_path:
                view.image_path = image_path
            scene.cameras[image_id] = view
        
        for point_id, (xyz, observations) in points3d.items():
            point = Point3D(
                point_id=point_id,
                xyz=xyz,
                observations=observations,
                is_outlier=False
            )
            scene.points3d[point_id] = point
        
        self.logger.info(f"读取了 {len(scene.cameras)} 个相机和 {len(scene.points3d)} 个3D点")
        return scene
    
    def _read_cameras(self) -> Dict[int, Tuple[str, int, int, List[float]]]:
        """读取相机参数
        
        Returns:
            相机参数字典 {camera_id: (model, width, height, params)}
        """
        cameras = {}
        
        with open(self.cameras_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = list(map(float, elems[4:]))
                
                cameras[camera_id] = (model, width, height, params)
        
        return cameras
    
    def _read_images(self, cameras: Dict[int, Tuple[str, int, int, List[float]]]) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """读取图像参数
        
        Args:
            cameras: 相机参数字典
            
        Returns:
            图像参数字典 {image_id: (K, R, t, image_path)}
        """
        images = {}
        
        with open(self.images_path, 'r') as f:
            lines = f.readlines()
            
            line_index = 0
            while line_index < len(lines):
                line = lines[line_index].strip()
                line_index += 1
                
                if line == "" or line.startswith("#"):
                    continue
                
                elems = line.split()
                image_id = int(elems[0])
                qw, qx, qy, qz = map(float, elems[1:5])
                tx, ty, tz = map(float, elems[5:8])
                camera_id = int(elems[8])
                image_name = " ".join(elems[9:])
                
                # 获取相机参数
                if camera_id not in cameras:
                    self.logger.warning(f"找不到相机 {camera_id} 的参数")
                    continue
                
                model, width, height, params = cameras[camera_id]
                
                # 构建内参矩阵
                K = np.eye(3)
                if model == "SIMPLE_PINHOLE":
                    K[0, 0] = K[1, 1] = params[0]  # f
                    K[0, 2] = params[1]  # cx
                    K[1, 2] = params[2]  # cy
                elif model == "PINHOLE":
                    K[0, 0] = params[0]  # fx
                    K[1, 1] = params[1]  # fy
                    K[0, 2] = params[2]  # cx
                    K[1, 2] = params[3]  # cy
                elif model == "SIMPLE_RADIAL":
                    K[0, 0] = K[1, 1] = params[0]  # f
                    K[0, 2] = params[1]  # cx
                    K[1, 2] = params[2]  # cy
                elif model == "RADIAL":
                    K[0, 0] = K[1, 1] = params[0]  # f
                    K[0, 2] = params[1]  # cx
                    K[1, 2] = params[2]  # cy
                
                # 四元数转旋转矩阵（世界到相机）
                R = self._quaternion_to_rotation_matrix(np.array([qw, qx, qy, qz]))
                
                # COLMAP存储的是相机中心C，需要转换为平移向量t
                # t = -R * C
                C = np.array([tx, ty, tz]).reshape(3, 1)
                t = -R @ C
                
                # 假设图像路径
                image_path = os.path.join(os.path.dirname(self.input_dir), "images", image_name)
                
                images[image_id] = (K, R, t, image_path)
                
                # 跳过点行
                line_index += 1
        
        return images
    
    def _read_points3d(self) -> Dict[int, Tuple[np.ndarray, Dict[int, int]]]:
        """读取3D点
        
        Returns:
            3D点字典 {point_id: (xyz, observations)}
        """
        points3d = {}
        
        with open(self.points3d_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                
                elems = line.split()
                point_id = int(elems[0])
                x, y, z = map(float, elems[1:4])
                
                # 解析观测轨迹
                observations = {}
                for i in range(8, len(elems), 2):
                    if i + 1 < len(elems):
                        image_id = int(elems[i])
                        keypoint_idx = int(elems[i + 1])
                        observations[image_id] = keypoint_idx
                
                points3d[point_id] = (np.array([x, y, z]), observations)
        
        return points3d
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """将四元数转换为旋转矩阵
        
        Args:
            q: 四元数 [w, x, y, z]
            
        Returns:
            3x3旋转矩阵
        """
        w, x, y, z = q
        
        R = np.zeros((3, 3))
        
        R[0, 0] = 1 - 2 * y * y - 2 * z * z
        R[0, 1] = 2 * x * y - 2 * w * z
        R[0, 2] = 2 * x * z + 2 * w * y
        
        R[1, 0] = 2 * x * y + 2 * w * z
        R[1, 1] = 1 - 2 * x * x - 2 * z * z
        R[1, 2] = 2 * y * z - 2 * w * x
        
        R[2, 0] = 2 * x * z - 2 * w * y
        R[2, 1] = 2 * y * z + 2 * w * x
        R[2, 2] = 1 - 2 * x * x - 2 * y * y
        
        return R