import math
import pyproj
import numpy as np

class GPSLocalConverter:
    _instance = None
    _is_initialized = False
    _ref_lat = None
    _ref_lon = None
    _ref_alt = 0.0
    _transformer = None
    
    @classmethod
    def initialize(cls, ref_lat, ref_lon, ref_alt=0.0):
        """初始化GPS坐标转换器
        
        Args:
            ref_lat: 参考纬度
            ref_lon: 参考经度
            ref_alt: 参考高度
        """
        cls._ref_lat = ref_lat
        cls._ref_lon = ref_lon
        cls._ref_alt = ref_alt
        cls._transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",   # WGS84 经纬度
            "EPSG:4978",   # WGS84 ECEF 笛卡尔
            always_xy=True
        )
        cls._is_initialized = True
    
    @classmethod
    def to_local_enu(cls, lat, lon, alt=0.0) -> np.ndarray:
        """经纬度 → 局部ENU坐标（米）
        
        Args:
            lat: 纬度
            lon: 经度
            alt: 高度
            
        Returns:
            局部ENU坐标 [east, north, up]
        """
        # 如果未初始化，使用第一次调用的坐标作为参考点
        if not cls._is_initialized:
            cls.initialize(lat, lon, alt)
            return np.zeros(3)  # 参考点返回零向量
        
        # ECEF坐标
        x0, y0, z0 = cls._transformer.transform(cls._ref_lon, cls._ref_lat, cls._ref_alt)
        x,  y,  z  = cls._transformer.transform(lon, lat, alt)

        dx, dy, dz = x - x0, y - y0, z - z0

        # 旋转到ENU
        sin_lat = math.sin(math.radians(cls._ref_lat))
        cos_lat = math.cos(math.radians(cls._ref_lat))
        sin_lon = math.sin(math.radians(cls._ref_lon))
        cos_lon = math.cos(math.radians(cls._ref_lon))

        east  = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return np.array([east, north, up])
    
    @classmethod
    def reset(cls):
        """重置GPS坐标转换器"""
        cls._is_initialized = False
        cls._ref_lat = None
        cls._ref_lon = None
        cls._ref_alt = 0.0
        cls._transformer = None