"""
图像加载模块

负责图像加载和GPS/EXIF/XMP元数据解析
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path
import cv2
from dataclasses import dataclass
import piexif
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import xml.etree.ElementTree as ET
from datetime import datetime

from core.data_models import ImageMeta


class ImageLoader:
    """图像加载器
    
    负责加载图像文件并解析其元数据
    """
    
    def __init__(self, image_dir: Optional[str] = None, max_image_size: int = 0):
        """初始化图像加载器
        
        Args:
            image_dir: 图像目录路径
            max_image_size: 图像最大尺寸（0表示不调整大小）
        """
        self.image_dir = image_dir
        self.max_image_size = max_image_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 支持的图像扩展名
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    def find_images(self) -> List[str]:
        """查找目录中所有支持的图像文件
        
        Returns:
            图像文件路径列表
        """
        image_paths = []
        
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                ext = os.path.splitext(file.lower())[1]
                if ext in self.supported_extensions:
                    image_paths.append(os.path.join(root, file))
        
        self.logger.info(f"在 {self.image_dir} 中找到 {len(image_paths)} 张图像")
        return image_paths
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载图像和元数据
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            (图像数据, 元数据字典)
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 调整图像大小（如果需要）
        if self.max_image_size > 0:
            h, w = image.shape[:2]
            if max(h, w) > self.max_image_size:
                scale = self.max_image_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                self.logger.debug(f"调整图像大小: {w}x{h} -> {new_size[0]}x{new_size[1]}")
        
        # 解析元数据
        metadata = self._parse_metadata(image_path)
        
        return image, metadata
    
    def load_directory(self, image_dir: str = None) -> List[ImageMeta]:
        """加载目录中的所有图像
        
        Args:
            image_dir: 图像目录路径，如果为None则使用初始化时的目录
            
        Returns:
            ImageMeta对象列表
        """
        if image_dir is not None:
            self.image_dir = image_dir
        
        image_paths = self.find_images()
        images = []
        
        for i, image_path in enumerate(image_paths):
            try:
                _, metadata = self.load_image(image_path)
                image_meta = self.create_image_meta(i, metadata)
                images.append(image_meta)
            except Exception as e:
                self.logger.warning(f"加载图像失败: {image_path}, 错误: {e}")
        
        return images

    def _parse_metadata(self, image_path: str) -> Dict[str, Any]:
        """解析图像元数据
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            元数据字典
        """
        metadata = {
            'image_path': image_path,
            'width': 0,
            'height': 0,
            'focal_length_px': 0.0,
            'cx': 0.0,
            'cy': 0.0,
            'k1': 0.0,
            'k2': 0.0,
            'camera_model': 'SIMPLE_RADIAL',
            'camera_serial': '',
            'gps_lat': None,
            'gps_lon': None,
            'gps_alt': None,
            'yaw': None,
            'pitch': None,
            'roll': None,
            'capture_time': None
        }
        
        try:
            # 使用PIL加载图像以获取EXIF数据
            with Image.open(image_path) as img:
                width, height = img.size
                metadata['width'] = width
                metadata['height'] = height
                
                # 获取EXIF数据
                exif_data = None
                try:
                    exif_data = piexif.load(img.info['exif']) if 'exif' in img.info else None
                except Exception as e:
                    self.logger.warning(f"无法加载EXIF数据: {e}")
                
                if exif_data:
                    # 提取相机序列号
                    if piexif.ExifIFD.BodySerialNumber in exif_data.get('Exif', {}):
                        metadata['camera_serial'] = exif_data['Exif'][piexif.ExifIFD.BodySerialNumber].decode('utf-8', errors='ignore')
                    sensor_width_mm = None
                    # 提取焦距
                    if piexif.ExifIFD.FocalLength in exif_data.get('Exif', {}):
                        focal_mm = exif_data['Exif'][piexif.ExifIFD.FocalLength]
                        if isinstance(focal_mm, tuple) and len(focal_mm) == 2:
                            focal_mm = focal_mm[0] / focal_mm[1]
                        
                        # 方法2: 使用35mm等效焦距估算，考虑宽高比
                        if sensor_width_mm is None and piexif.ExifIFD.FocalLengthIn35mmFilm in exif_data.get('Exif', {}):
                            focal_35mm = exif_data['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm]
                            if focal_35mm > 0 and focal_mm > 0:
                                # 计算图像的宽高比
                                aspect_ratio = width / height if height > 0 else 1.5
                                
                                # 35mm全画幅的对角线长度（约43.3mm）
                                full_frame_diagonal = 43.3
                                
                                # 根据35mm等效焦距比例计算传感器对角线长度
                                sensor_diagonal = full_frame_diagonal * focal_mm / focal_35mm
                                
                                # 根据宽高比计算传感器宽度
                                # 对角线^2 = 宽^2 + 高^2
                                # 宽 = 对角线 / sqrt(1 + 1/宽高比^2)
                                sensor_width_mm = sensor_diagonal / np.sqrt(1 + 1/(aspect_ratio**2))
                                
                                self.logger.debug(f"通过35mm等效焦距估算传感器宽度: {sensor_width_mm:.2f}mm, "f"宽高比: {aspect_ratio:.2f}, 对角线: {sensor_diagonal:.2f}mm")
                        
                        # 计算像素焦距
                        metadata['focal_length_px'] = focal_mm * width / sensor_width_mm
                    
                    # 提取GPS信息
                    if 'GPS' in exif_data and exif_data['GPS']:
                        gps_data = exif_data['GPS']
                        
                        # 提取纬度
                        if piexif.GPSIFD.GPSLatitude in gps_data and piexif.GPSIFD.GPSLatitudeRef in gps_data:
                            lat = self._convert_to_degrees(gps_data[piexif.GPSIFD.GPSLatitude])
                            if gps_data[piexif.GPSIFD.GPSLatitudeRef] == b'S':
                                lat = -lat
                            metadata['gps_lat'] = lat
                        
                        # 提取经度
                        if piexif.GPSIFD.GPSLongitude in gps_data and piexif.GPSIFD.GPSLongitudeRef in gps_data:
                            lon = self._convert_to_degrees(gps_data[piexif.GPSIFD.GPSLongitude])
                            if gps_data[piexif.GPSIFD.GPSLongitudeRef] == b'W':
                                lon = -lon
                            metadata['gps_lon'] = lon
                        
                        # 提取高度
                        if piexif.GPSIFD.GPSAltitude in gps_data:
                            alt_tuple = gps_data[piexif.GPSIFD.GPSAltitude]
                            if isinstance(alt_tuple, tuple) and len(alt_tuple) == 2 and alt_tuple[1] > 0:
                                alt = alt_tuple[0] / alt_tuple[1]
                                if piexif.GPSIFD.GPSAltitudeRef in gps_data and gps_data[piexif.GPSIFD.GPSAltitudeRef] == 1:
                                    alt = -alt
                                metadata['gps_alt'] = alt
                    
                    # 提取拍摄时间
                    if piexif.ExifIFD.DateTimeOriginal in exif_data.get('Exif', {}):
                        date_str = exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore')
                        try:
                            dt = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                            metadata['capture_time'] = dt.isoformat()
                        except ValueError:
                            pass
            
            # 尝试从XMP数据中提取姿态信息（常见于无人机影像）
            self._parse_xmp_metadata(image_path, metadata)
            
            # 设置主点坐标（如果没有，假设在图像中心）
            if metadata['cx'] == 0.0:
                metadata['cx'] = width / 2.0
            if metadata['cy'] == 0.0:
                metadata['cy'] = height / 2.0
            
            # 如果没有有效的焦距，使用经验公式（约等于对角线长度）
            if metadata['focal_length_px'] <= 0.0:
                metadata['focal_length_px'] = max(width, height) * 1.2
            
        except Exception as e:
            self.logger.error(f"解析图像元数据时出错: {e}")
        
        return metadata
    
    def _parse_xmp_metadata(self, image_path: str, metadata: Dict[str, Any]) -> None:
        """解析XMP元数据（主要用于无人机姿态信息）
        
        Args:
            image_path: 图像文件路径
            metadata: 要更新的元数据字典
        """
        try:
            # 读取文件的前10KB来查找XMP数据
            with open(image_path, 'rb') as f:
                data = f.read(10240)
            
            # 查找XMP数据块
            xmp_start = data.find(b'<x:xmpmeta')
            if xmp_start == -1:
                return
            
            xmp_end = data.find(b'</x:xmpmeta>', xmp_start)
            if xmp_end == -1:
                return
            
            xmp_data = data[xmp_start:xmp_end + 12].decode('utf-8', errors='ignore')
            
            # 解析XMP XML
            try:
                root = ET.fromstring(xmp_data)
                
                # DJI无人机XMP命名空间
                namespaces = {
                    'drone-dji': 'http://www.dji.com/drone-dji/1.0/',
                    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                    'x': 'adobe:ns:meta/'
                }
                
                # 尝试提取DJI无人机姿态数据
                for ns in namespaces:
                    # 提取偏航角（航向）
                    yaw_elem = root.find(f".//drone-dji:FlightYawDegree", namespaces)
                    if yaw_elem is not None and yaw_elem.text:
                        metadata['yaw'] = float(yaw_elem.text)
                    
                    # 提取俯仰角
                    pitch_elem = root.find(f".//drone-dji:FlightPitchDegree", namespaces)
                    if pitch_elem is not None and pitch_elem.text:
                        metadata['pitch'] = float(pitch_elem.text)
                    
                    # 提取横滚角
                    roll_elem = root.find(f".//drone-dji:FlightRollDegree", namespaces)
                    if roll_elem is not None and roll_elem.text:
                        metadata['roll'] = float(roll_elem.text)
            
            except ET.ParseError:
                self.logger.warning(f"解析XMP数据时出错: {image_path}")
        
        except Exception as e:
            self.logger.warning(f"读取XMP元数据时出错: {e}")
    
    def _convert_to_degrees(self, value: tuple) -> float:
        """将EXIF GPS坐标转换为度
        
        Args:
            value: EXIF GPS坐标元组 ((度分子,度分母), (分分子,分分母), (秒分子,秒分母))
            
        Returns:
            十进制度数
        """
        try:
            d = value[0][0] / value[0][1] if value[0][1] > 0 else 0
            m = value[1][0] / value[1][1] if value[1][1] > 0 else 0
            s = value[2][0] / value[2][1] if value[2][1] > 0 else 0
            
            return d + (m / 60.0) + (s / 3600.0)
        except (IndexError, ZeroDivisionError, TypeError):
            return 0.0
    
    def create_image_meta(self, image_id: int, metadata: Dict[str, Any]) -> ImageMeta:
        """从元数据创建ImageMeta对象
        
        Args:
            image_id: 图像ID
            metadata: 元数据字典
            
        Returns:
            ImageMeta对象
        """
        return ImageMeta(
            image_id=image_id,
            image_path=metadata['image_path'],
            width=metadata['width'],
            height=metadata['height'],
            focal_length_px=metadata['focal_length_px'],
            cx=metadata['cx'],
            cy=metadata['cy'],
            k1=metadata['k1'],
            k2=metadata['k2'],
            camera_model=metadata['camera_model'],
            camera_serial=metadata['camera_serial'],
            gps_lat=metadata['gps_lat'],
            gps_lon=metadata['gps_lon'],
            gps_alt=metadata['gps_alt'],
            yaw=metadata['yaw'],
            pitch=metadata['pitch'],
            roll=metadata['roll'],
            capture_time=metadata['capture_time']
        )