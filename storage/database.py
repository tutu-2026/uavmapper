"""
数据库访问模块

提供统一的数据库访问接口，用于存储和检索特征、匹配和几何验证结果
对标COLMAP数据库结构，使用SQLite实现
"""
import os
import sqlite3
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np

from core.data_models import ImageMeta, FeatureData, MatchData, VerificationResult, GeometryModel


class Database:
    """数据库访问类
    
    提供对SQLite数据库的统一访问接口，用于存储和检索特征、匹配和几何验证结果
    """
    
    def __init__(self, database_path: str):
        """初始化数据库
        
        Args:
            database_path: 数据库文件路径
        """
        self.database_path = database_path
        self.connection = None
        
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
        
        # 连接数据库并初始化表结构
        self._connect()
        self._create_schema()
    

    def __del__(self):
        """析构函数，确保关闭数据库连接"""
        self.close()
    

    def _connect(self) -> None:
        """连接到SQLite数据库"""
        self.connection = sqlite3.connect(self.database_path)
        # 启用外键约束
        self.connection.execute("PRAGMA foreign_keys = ON")
    

    def close(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
    

    def _create_schema(self) -> None:
        """创建数据库表结构"""
        cursor = self.connection.cursor()
        
        # 相机表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY,
            model TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            focal_length_px REAL NOT NULL,
            cx REAL NOT NULL,
            cy REAL NOT NULL,
            k1 REAL NOT NULL,
            k2 REAL NOT NULL,
            camera_serial TEXT
        )
        """)
        
        # 图像表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY,
            camera_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            gps_lat REAL,
            gps_lon REAL,
            gps_alt REAL,
            yaw REAL,
            pitch REAL,
            roll REAL,
            capture_time TEXT,
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        )
        """)
        
        # 特征点表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY,
            num_keypoints INTEGER NOT NULL,
            keypoints_xy BLOB NOT NULL,
            scales BLOB NOT NULL,
            orientations BLOB NOT NULL,
            scores BLOB NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id)
        )
        """)
        
        # 描述符表（与keypoints分表，BA阶段可跳过加载）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY,
            descriptors BLOB NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id)
        )
        """)
        
        # 匹配表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY,
            img_id1 INTEGER NOT NULL,
            img_id2 INTEGER NOT NULL,
            matches BLOB NOT NULL,
            scores BLOB NOT NULL,
            FOREIGN KEY(img_id1) REFERENCES images(image_id),
            FOREIGN KEY(img_id2) REFERENCES images(image_id)
        )
        """)
        
        # 两视图几何表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY,
            img_id1 INTEGER NOT NULL,
            img_id2 INTEGER NOT NULL,
            is_valid INTEGER NOT NULL,
            model_type TEXT NOT NULL,
            E_mat BLOB,
            F_mat BLOB,
            H_mat BLOB,
            inlier_mask BLOB,
            inlier_count INTEGER NOT NULL,
            inlier_ratio REAL NOT NULL,
            relative_R BLOB,
            relative_t BLOB,
            parallax_score REAL,
            FOREIGN KEY(img_id1) REFERENCES images(image_id),
            FOREIGN KEY(img_id2) REFERENCES images(image_id)
        )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_camera_id ON images(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_gps ON images(gps_lat, gps_lon)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_img_id1 ON matches(img_id1)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_img_id2 ON matches(img_id2)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_two_view_geometries_is_valid ON two_view_geometries(is_valid)")
        
        self.connection.commit()
    

    def _get_pair_id(self, id1: int, id2: int) -> int:
        """计算图像对的唯一ID
        
        确保id1 < id2，并生成唯一的pair_id
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            
        Returns:
            唯一的pair_id
        """
        if id1 > id2:
            id1, id2 = id2, id1
        return id1 * 1000000 + id2
    

    def save_camera(self, meta: ImageMeta) -> int:
        """保存相机信息到数据库
        
        如果相同参数的相机已存在，则返回现有的camera_id
        
        Args:
            meta: 图像元数据
            
        Returns:
            camera_id
        """
        cursor = self.connection.cursor()
        
        # 检查是否已存在相同参数的相机
        cursor.execute("""
        SELECT camera_id FROM cameras
        WHERE model=? AND width=? AND height=? AND focal_length_px=? AND cx=? AND cy=? AND k1=? AND k2=?
        """, (meta.camera_model, meta.width, meta.height, meta.focal_length_px, 
              meta.cx, meta.cy, meta.k1, meta.k2))
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # 插入新相机
        cursor.execute("""
        INSERT INTO cameras (model, width, height, focal_length_px, cx, cy, k1, k2, camera_serial)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (meta.camera_model, meta.width, meta.height, meta.focal_length_px, 
              meta.cx, meta.cy, meta.k1, meta.k2, meta.camera_serial))
        
        camera_id = cursor.lastrowid
        self.connection.commit()
        return camera_id
    

    def load_camera(self, camera_id: int) -> Dict[str, Any]:
        """加载相机信息
        
        Args:
            camera_id: 相机ID
            
        Returns:
            相机参数字典
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT model, width, height, focal_length_px, cx, cy, k1, k2, camera_serial
        FROM cameras WHERE camera_id=?
        """, (camera_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Camera with ID {camera_id} not found")
        
        return {
            "camera_id": camera_id,
            "model": result[0],
            "width": result[1],
            "height": result[2],
            "focal_length_px": result[3],
            "cx": result[4],
            "cy": result[5],
            "k1": result[6],
            "k2": result[7],
            "camera_serial": result[8]
        }
    

    def save_image(self, meta: ImageMeta) -> None:
        """保存图像信息到数据库
        
        Args:
            meta: 图像元数据
        """
        cursor = self.connection.cursor()
        
        # 获取或创建相机ID
        camera_id = self.save_camera(meta)
        
        # 插入图像
        cursor.execute("""
        INSERT OR REPLACE INTO images 
        (image_id, camera_id, image_path, gps_lat, gps_lon, gps_alt, yaw, pitch, roll, capture_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (meta.image_id, camera_id, meta.image_path, meta.gps_lat, meta.gps_lon, meta.gps_alt,
              meta.yaw, meta.pitch, meta.roll, meta.capture_time))
        
        self.connection.commit()
    

    def load_image_meta(self, image_id: int) -> ImageMeta:
        """加载图像信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            图像元数据
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT i.image_id, i.image_path, c.width, c.height, c.focal_length_px, c.cx, c.cy,
               c.k1, c.k2, c.model, c.camera_serial, i.gps_lat, i.gps_lon, i.gps_alt,
               i.yaw, i.pitch, i.roll, i.capture_time
        FROM images i JOIN cameras c ON i.camera_id = c.camera_id
        WHERE i.image_id=?
        """, (image_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Image with ID {image_id} not found")
        
        return ImageMeta(
            image_id=result[0],
            image_path=result[1],
            width=result[2],
            height=result[3],
            focal_length_px=result[4],
            cx=result[5],
            cy=result[6],
            k1=result[7],
            k2=result[8],
            camera_model=result[9],
            camera_serial=result[10],
            gps_lat=result[11],
            gps_lon=result[12],
            gps_alt=result[13],
            yaw=result[14],
            pitch=result[15],
            roll=result[16],
            capture_time=result[17]
        )
    

    def load_all_images(self) -> List[ImageMeta]:
        """加载所有图像信息
        
        Returns:
            图像元数据列表
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT i.image_id, i.image_path, c.width, c.height, c.focal_length_px, c.cx, c.cy,
               c.k1, c.k2, c.model, c.camera_serial, i.gps_lat, i.gps_lon, i.gps_alt,
               i.yaw, i.pitch, i.roll, i.capture_time
        FROM images i JOIN cameras c ON i.camera_id = c.camera_id
        """)
        
        results = cursor.fetchall()
        images = []
        
        for result in results:
            images.append(ImageMeta(
                image_id=result[0],
                image_path=result[1],
                width=result[2],
                height=result[3],
                focal_length_px=result[4],
                cx=result[5],
                cy=result[6],
                k1=result[7],
                k2=result[8],
                camera_model=result[9],
                camera_serial=result[10],
                gps_lat=result[11],
                gps_lon=result[12],
                gps_alt=result[13],
                yaw=result[14],
                pitch=result[15],
                roll=result[16],
                capture_time=result[17]
            ))
        
        return images
    

    def load_images_in_gps_bbox(self, min_lat: float, min_lon: float, 
                               max_lat: float, max_lon: float) -> List[int]:
        """加载指定GPS范围内的图像ID
        
        Args:
            min_lat: 最小纬度
            min_lon: 最小经度
            max_lat: 最大纬度
            max_lon: 最大经度
            
        Returns:
            图像ID列表
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT image_id FROM images
        WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL
          AND gps_lat BETWEEN ? AND ?
          AND gps_lon BETWEEN ? AND ?
        """, (min_lat, max_lat, min_lon, max_lon))
        
        results = cursor.fetchall()
        return [result[0] for result in results]
    

    def save_keypoints(self, data: FeatureData) -> None:
        """保存特征点信息（不含描述符）
        
        Args:
            data: 特征数据
        """
        cursor = self.connection.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO keypoints 
        (image_id, num_keypoints, keypoints_xy, scales, orientations, scores)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            data.image_id,
            data.num_keypoints,
            data.keypoints_xy.tobytes(),
            data.scales.tobytes(),
            data.orientations.tobytes(),
            data.scores.tobytes()
        ))
        
        self.connection.commit()
    

    def save_descriptors(self, image_id: int, descriptors: np.ndarray) -> None:
        """单独保存描述符
        
        Args:
            image_id: 图像ID
            descriptors: 描述符数组
        """
        cursor = self.connection.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO descriptors 
        (image_id, descriptors)
        VALUES (?, ?)
        """, (
            image_id,
            descriptors.tobytes()
        ))
        
        self.connection.commit()
    

    def load_keypoints(self, image_id: int, load_desc: bool = False) -> FeatureData:
        """加载特征点信息
        
        Args:
            image_id: 图像ID
            load_desc: 是否加载描述符
            
        Returns:
            特征数据
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT num_keypoints, keypoints_xy, scales, orientations, scores
        FROM keypoints WHERE image_id=?
        """, (image_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Keypoints for image {image_id} not found")
        
        num_keypoints = result[0]
        keypoints_xy = np.frombuffer(result[1], dtype=np.float32).reshape(-1, 2)
        scales = np.frombuffer(result[2], dtype=np.float32)
        orientations = np.frombuffer(result[3], dtype=np.float32)
        scores = np.frombuffer(result[4], dtype=np.float32)
        
        descriptors = None
        if load_desc:
            cursor.execute("""
            SELECT descriptors FROM descriptors WHERE image_id=?
            """, (image_id,))
            
            desc_result = cursor.fetchone()
            if desc_result:
                descriptors = np.frombuffer(desc_result[0], dtype=np.float32).reshape(num_keypoints, -1)
        
        return FeatureData(
            image_id=image_id,
            num_keypoints=num_keypoints,
            keypoints_xy=keypoints_xy,
            scales=scales,
            orientations=orientations,
            scores=scores,
            descriptors=descriptors if descriptors is not None else np.array([], dtype=np.float32)
        )
    

    def save_matches(self, data: MatchData) -> None:
        """保存匹配信息
        
        Args:
            data: 匹配数据
        """
        # 确保id1 < id2
        img_id1, img_id2 = data.img_id1, data.img_id2
        matches = data.matches
        scores = data.scores
        
        if img_id1 > img_id2:
            img_id1, img_id2 = img_id2, img_id1
            # 交换匹配对的列
            matches = np.fliplr(matches)
        
        pair_id = self._get_pair_id(img_id1, img_id2)
        
        cursor = self.connection.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO matches 
        (pair_id, img_id1, img_id2, matches, scores)
        VALUES (?, ?, ?, ?, ?)
        """, (
            pair_id,
            img_id1,
            img_id2,
            matches.tobytes(),
            scores.tobytes()
        ))
        
        self.connection.commit()
    

    def load_matches(self, id1: int, id2: int) -> MatchData:
        """加载匹配信息
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            
        Returns:
            匹配数据
        """
        # 确保id1 < id2
        original_order = id1 < id2
        if not original_order:
            id1, id2 = id2, id1
        
        pair_id = self._get_pair_id(id1, id2)
        
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT matches, scores FROM matches WHERE pair_id=?
        """, (pair_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Matches for image pair ({id1}, {id2}) not found")
        
        matches_blob = result[0]
        scores_blob = result[1]

        matches = np.frombuffer(result[0], dtype=np.int64).reshape(-1, 2)
        scores = np.frombuffer(result[1], dtype=np.float32)
        
        # 如果原始顺序是反的，需要交换匹配对的列
        if not original_order:
            matches = np.fliplr(matches)
            id1, id2 = id2, id1
        
        return MatchData(
            img_id1=id1,
            img_id2=id2,
            matches=matches,
            scores=scores
        )
    
    
    def load_camera_intrinsics(self, image_id: int) -> Optional[np.ndarray]:
        """加载图像的相机内参矩阵
        
        Args:
            image_id: 图像ID
            
        Returns:
            相机内参矩阵 [3, 3]，如果没有足够信息则返回None
        """
        try:
            # 加载图像元数据
            meta = self.load_image_meta(image_id)
            
            # 检查是否有必要的内参信息
            if (meta.focal_length_px is None or meta.width is None or 
                meta.height is None or meta.cx is None or meta.cy is None):
                return None
            
            # 构建相机内参矩阵
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = meta.focal_length_px  # fx
            K[1, 1] = meta.focal_length_px  # fy (假设像素是正方形的)
            K[0, 2] = meta.cx               # cx (主点x坐标)
            K[1, 2] = meta.cy               # cy (主点y坐标)
            
            return K
            
        except Exception as e:
            self.logger.warning(f"加载图像 {image_id} 的相机内参失败: {str(e)}")
            return None


    def load_all_pairs(self) -> List[Tuple[int, int]]:
        """加载所有匹配对
        
        Returns:
            (img_id1, img_id2)列表，保证id1 < id2
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT img_id1, img_id2 FROM matches")
        
        results = cursor.fetchall()
        return [(result[0], result[1]) for result in results]
    

    def save_verification(self, result: VerificationResult) -> None:
        """保存几何验证结果
        
        Args:
            result: 验证结果
        """
        # 确保id1 < id2
        img_id1, img_id2 = result.img_id1, result.img_id2
        if img_id1 > img_id2:
            img_id1, img_id2 = img_id2, img_id1
        
        pair_id = self._get_pair_id(img_id1, img_id2)
        
        # 转换布尔掩码为字节
        inlier_mask_bytes = result.inlier_mask.tobytes() if result.inlier_mask is not None else None
        
        # 转换矩阵为字节
        E_mat_bytes = result.E_mat.tobytes() if result.E_mat is not None else None
        F_mat_bytes = result.F_mat.tobytes() if result.F_mat is not None else None
        H_mat_bytes = result.H_mat.tobytes() if result.H_mat is not None else None
        relative_R_bytes = result.relative_R.tobytes() if result.relative_R is not None else None
        relative_t_bytes = result.relative_t.tobytes() if result.relative_t is not None else None
        
        cursor = self.connection.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO two_view_geometries 
        (pair_id, img_id1, img_id2, is_valid, model_type, E_mat, F_mat, H_mat, 
         inlier_mask, inlier_count, inlier_ratio, relative_R, relative_t, parallax_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair_id,
            img_id1,
            img_id2,
            1 if result.is_valid else 0,
            result.model_type.value,
            E_mat_bytes,
            F_mat_bytes,
            H_mat_bytes,
            inlier_mask_bytes,
            result.inlier_count,
            result.inlier_ratio,
            relative_R_bytes,
            relative_t_bytes,
            result.parallax_score
        ))
        
        self.connection.commit()
    

    def load_verification(self, id1: int, id2: int) -> VerificationResult:
        """加载几何验证结果
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            
        Returns:
            验证结果
        """
        # 确保id1 < id2
        original_order = id1 < id2
        if not original_order:
            id1, id2 = id2, id1
        
        pair_id = self._get_pair_id(id1, id2)
        
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT is_valid, model_type, E_mat, F_mat, H_mat, inlier_mask, 
               inlier_count, inlier_ratio, relative_R, relative_t, parallax_score
        FROM two_view_geometries WHERE pair_id=?
        """, (pair_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Verification result for image pair ({id1}, {id2}) not found")
        
        is_valid = bool(result[0])
        model_type = GeometryModel(result[1])
        
        # 解析矩阵
        E_mat = np.frombuffer(result[2], dtype=np.float64).reshape(3, 3) if result[2] else None
        F_mat = np.frombuffer(result[3], dtype=np.float64).reshape(3, 3) if result[3] else None
        H_mat = np.frombuffer(result[4], dtype=np.float64).reshape(3, 3) if result[4] else None
        
        # 解析掩码
        inlier_mask = np.frombuffer(result[5], dtype=bool) if result[5] else None
        
        inlier_count = result[6]
        inlier_ratio = result[7]
        
        # 解析相对位姿
        relative_R = np.frombuffer(result[8], dtype=np.float64).reshape(3, 3) if result[8] else None
        relative_t = np.frombuffer(result[9], dtype=np.float64) if result[9] else None
        
        parallax_score = result[10]
        
        # 如果原始顺序是反的，需要调整相对位姿
        if not original_order and relative_R is not None and relative_t is not None:
            # 对于反向的相对位姿，R' = R^T, t' = -R^T * t
            relative_R = relative_R.T
            relative_t = -np.dot(relative_R, relative_t)
            id1, id2 = id2, id1
        
        return VerificationResult(
            img_id1=id1,
            img_id2=id2,
            is_valid=is_valid,
            model_type=model_type,
            E_mat=E_mat,
            F_mat=F_mat,
            H_mat=H_mat,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            relative_R=relative_R,
            relative_t=relative_t,
            parallax_score=parallax_score
        )
    

    def load_valid_pairs(self) -> List[Tuple[int, int]]:
        """加载所有有效的几何验证对
        
        Returns:
            (img_id1, img_id2)列表，保证id1 < id2
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        SELECT img_id1, img_id2 FROM two_view_geometries WHERE is_valid=1
        """)
        
        results = cursor.fetchall()
        return [(result[0], result[1]) for result in results]
    

    def load_inlier_matches(self, id1: int, id2: int) -> Optional[MatchData]:
        """加载几何验证后的内点匹配
        
        结合matches表和two_view_geometries表，返回几何验证后的内点匹配
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            
        Returns:
            内点匹配数据，如果验证结果不存在或无效则返回None
        """
        try:
            # 1. 尝试加载几何验证结果
            verification = self.load_verification(id1, id2)
            
            # 2. 检查验证结果是否有效
            if not verification.is_valid or verification.inlier_count == 0:
                return None
            
            # 3. 加载原始匹配
            match_data = self.load_matches(id1, id2)
            
            # 4. 使用内点掩码过滤匹配
            inlier_mask = verification.inlier_mask
            inlier_matches = match_data.matches[inlier_mask]
            
            # 5. 如果有分数，也进行过滤
            if match_data.scores is not None and len(match_data.scores) == len(match_data.matches):
                inlier_scores = match_data.scores[inlier_mask]
            else:
                inlier_scores = np.ones(len(inlier_matches), dtype=np.float32)
            
            # 6. 创建并返回内点匹配数据
            return MatchData(
                img_id1=id1,
                img_id2=id2,
                matches=inlier_matches,
                scores=inlier_scores
            )
        
        except Exception as e:
            # 如果任何步骤出错（例如验证结果或匹配不存在），返回None
            return None