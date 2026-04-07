"""
工作空间管理模块

管理工作空间目录、数据库连接和运行时状态
"""
import os
import json
import pickle
from typing import Dict, List, Optional, Any
import logging

from config import WorkspaceConfig
from storage.database import Database
from core.scene import SceneManager


class Workspace:
    """工作空间管理类
    
    管理工作空间目录、数据库连接和运行时状态
    """
    
    def __init__(self, config: WorkspaceConfig):
        """初始化工作空间
        
        Args:
            config: 工作空间配置
        """
        self.config = config
        self.logger = logging.getLogger("Workspace")
        
        # 确保工作空间目录存在
        os.makedirs(config.workspace_dir, exist_ok=True)
        
        # 确保导出目录存在
        if config.export_dir:
            os.makedirs(config.export_dir, exist_ok=True)
        
        # 数据库路径
        if not config.database_path:
            self.database_path = os.path.join(config.workspace_dir, "database.db")
        else:
            self.database_path = config.database_path
        
        # 场景状态路径
        self.scene_path = os.path.join(config.workspace_dir, "scene.pkl")
        
        # 摘要路径
        self.summary_path = os.path.join(config.workspace_dir, "summary.json")
        
        # 数据库实例
        self._db = None
    
    def get_database(self) -> Database:
        """获取数据库实例
        
        Returns:
            数据库实例
        """
        if self._db is None:
            self._db = Database(self.database_path)
        return self._db
    
    def close(self) -> None:
        """关闭工作空间，释放资源"""
        if self._db:
            self._db.close()
            self._db = None
    
    def save_scene(self, scene: SceneManager) -> None:
        """保存场景状态
        
        Args:
            scene: 场景管理器实例
        """
        with open(self.scene_path, 'wb') as f:
            pickle.dump(scene, f)
        
        # 同时保存摘要信息
        self._save_summary(scene)
        
        self.logger.info(f"场景已保存到 {self.scene_path}")
    
    def load_scene(self) -> SceneManager:
        """加载场景状态
        
        Returns:
            场景管理器实例
        """
        if not os.path.exists(self.scene_path):
            self.logger.warning(f"场景文件不存在: {self.scene_path}，返回空场景")
            return SceneManager()
        
        with open(self.scene_path, 'rb') as f:
            scene = pickle.load(f)
        
        self.logger.info(f"已从 {self.scene_path} 加载场景")
        return scene
    
    def _save_summary(self, scene: SceneManager) -> None:
        """保存重建摘要信息
        
        Args:
            scene: 场景管理器实例
        """
        stats = scene.get_statistics()
        
        # 计算有效点的平均观测数
        valid_points = [p for p in scene.points3d.values() if not p.is_outlier]
        if valid_points:
            avg_observations = sum(len(p.observations) for p in valid_points) / len(valid_points)
        else:
            avg_observations = 0
        
        # 计算已注册相机比例
        db = self.get_database()
        total_images = len(db.load_all_images())
        registered_ratio = stats["registered_cameras"] / total_images if total_images > 0 else 0
        
        summary = {
            "total_images": total_images,
            "registered_cameras": stats["registered_cameras"],
            "registration_ratio": registered_ratio,
            "total_points": stats["total_points"],
            "valid_points": stats["valid_points"],
            "avg_observations": avg_observations,
            "workspace_path": self.config.workspace_dir,
            "database_path": self.database_path,
            "scene_path": self.scene_path
        }
        
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"摘要已保存到 {self.summary_path}")
    
    def load_summary(self) -> Dict[str, Any]:
        """加载重建摘要信息
        
        Returns:
            摘要信息字典
        """
        if not os.path.exists(self.summary_path):
            self.logger.warning(f"摘要文件不存在: {self.summary_path}")
            return {}
        
        with open(self.summary_path, 'r') as f:
            summary = json.load(f)
        
        return summary
    
    def get_image_path(self, image_name: str) -> str:
        """获取图像的完整路径
        
        Args:
            image_name: 图像文件名
            
        Returns:
            图像的完整路径
        """
        return os.path.join(self.config.image_dir, image_name)
    
    def get_export_path(self, filename: str) -> str:
        """获取导出文件的完整路径
        
        Args:
            filename: 导出文件名
            
        Returns:
            导出文件的完整路径
        """
        if self.config.export_dir:
            return os.path.join(self.config.export_dir, filename)
        else:
            return os.path.join(self.config.workspace_dir, "exports", filename)