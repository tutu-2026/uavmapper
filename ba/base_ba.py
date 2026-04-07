"""
集束调整基类模块

定义集束调整的抽象基类
"""
import logging
from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod

from config import BAConfig
from core.scene import SceneManager


class BundleAdjusterBase(ABC):
    """集束调整基类
    
    所有BA实现的抽象基类
    """
    
    def __init__(self, config: BAConfig):
        """初始化集束调整器
        
        Args:
            config: BA配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def optimize(self, scene: SceneManager, **kwargs) -> Dict[str, Any]:
        """执行集束调整优化
        
        Args:
            scene: 场景管理器
            **kwargs: 额外参数
            
        Returns:
            包含优化统计信息的字典
        """
        pass