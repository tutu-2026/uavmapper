"""
全局参数配置模块

包含特征提取、匹配、BA和SFM流程的所有配置参数
"""
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class FeatureConfig:
    """特征提取与匹配相关配置"""
    # SIFT特征参数
    num_features: int = 20000                # 每张图像最多提取的特征点数量
    first_octave: int = 0                   # SIFT第一个八度层
    n_octave_layers: int = 3                # 每个八度层的层数
    contrast_threshold: float = 0.04        # 对比度阈值
    edge_threshold: float = 10.0            # 边缘响应阈值
    sigma: float = 1.6                      # 高斯模糊初始sigma
    use_rootsift: bool = True               # 是否使用RootSIFT归一化
    downsample_factor = 1                   # 图像降采样因子 (1表示不降采样)

    # 批处理参数
    batch_size: int = 1                     # GPU批处理大小
    
    # 匹配参数
    lowes_ratio: float = 0.8                # Lowe's比率测试阈值
    mutual_best_match: bool = False         # 是否使用双向最佳匹配
    max_num_matches: int = 8192             # 每对图像最大匹配数
    
    # 基于GPS的匹配候选对筛选
    use_gps_filtering: bool = True          # 是否使用GPS预筛选
    max_gps_distance: float = 100.0         # 最大GPS距离(米)，超过此距离不匹配
    min_matches_pairs: int = 50             # 每张图像至少匹配的图像数
    
    # FAISS匹配参数
    faiss_gpu_id: int = 0                   # FAISS使用的GPU ID
    faiss_nlist: int = 128                  # FAISS聚类中心数量
    faiss_nprobe: int = 32                  # FAISS搜索的聚类中心数量
    
    # 几何验证参数
    min_inlier_ratio: float = 0.1           # 最小内点比例
    min_inlier_count: int = 15              # 最小内点数量
    magsac_confidence: float = 0.999        # MAGSAC置信度
    magsac_max_iters: int = 10000           # MAGSAC最大迭代次数
    magsac_threshold: float = 1.0           # MAGSAC阈值(像素)
    
    ransac_threshold = 1.0        # RANSAC重投影阈值 (像素)
    ransac_confidence = 0.999     # RANSAC置信度
    ransac_max_iters = 10000      # RANSAC最大迭代次数
    min_inlier_ratio = 0.1        # 最小内点比例
    min_num_inliers = 15          # 最小内点数量

@dataclass
class BAConfig:
    """集束调整相关配置"""
    # 局部BA配置
    local_ba_window_size: int = 15          # 局部BA窗口大小
    local_ba_fixed_frames: int = 2          # 固定帧数量(解决gauge freedom)
    
    # 中等BA配置
    medium_ba_trigger_freq: int = 20        # 中等BA触发频率(每N帧)
    medium_ba_window_size: int = 100        # 中等BA窗口大小
    
    # 全局BA配置
    global_ba_trigger_freq: int = 100       # 全局BA触发频率(每N帧)
    
    # 优化器配置
    max_iterations: int = 200               # 最大迭代次数
    function_tolerance: float = 1e-6        # 函数容差
    gradient_tolerance: float = 1e-10       # 梯度容差
    parameter_tolerance: float = 1e-8       # 参数容差
    linear_solver_type: str = "DENSE_SCHUR" # 线性求解器类型
    loss_function_type: str = "HUBER"       # 损失函数类型
    huber_loss_parameter: float = 2.0       # Huber损失参数
    
    # 重投影误差阈值
    max_reproj_error: float = 4.0           # 最大重投影误差(像素)
    
    # GPS先验权重
    use_gps_prior: bool = False             # 是否使用GPS先验
    gps_prior_weight: float = 1.0           # GPS先验权重


@dataclass
class SFMConfig:
    """SFM流程相关配置"""
    # 初始化参数
    init_required_min_angle: float = 10      # 初始化最小角度
    init_limit_max_angle: float = 40         # 初始化最大限制角度
    init_optimal_angle: float = 15           # 初始化最优角度
    init_inlier_norm_ref: float = 300.0      # 内点数归一化参考
    
    # 三角化参数
    min_triangulation_angle_deg: float = 3.0   # 最小三角化角度(度)
    max_triangulation_angle_deg: float = 60.0  # 最大三角化角度(度)
    min_track_length: int = 2                  # 最小轨迹长度(观测数)
    
    # 增量注册参数
    min_pnp_inliers: int = 15               # PnP最小内点数
    min_pnp_inlier_ratio: float = 0.3       # PnP最小内点比例
    pnp_ransac_threshold: float = 4.0       # PnP RANSAC阈值(像素)
    
    # 增量注册顺序策略
    next_image_selection: Literal["most_observed", "best_score"] = "most_observed"
    
    # 过滤参数
    filter_max_reproj_error: float = 5.0    # 过滤最大重投影误差(像素)
    filter_min_track_length: int = 2        # 过滤最小轨迹长度


@dataclass
class WorkspaceConfig:
    """工作空间相关配置"""
    # 路径配置
    image_dir: str                          # 图像目录
    workspace_dir: str                      # 工作空间目录
    database_path: Optional[str] = None     # 数据库路径(默认为workspace_dir/database.db)
    export_dir: Optional[str] = None        # 导出目录(默认为workspace_dir/exports)
    
    # 导出选项
    export_colmap: bool = True              # 是否导出COLMAP格式
    export_ply: bool = True                 # 是否导出PLY点云
    
    # 系统配置
    num_cpu_threads: int = 4                # CPU线程数
    gpu_id: int = 0                         # GPU ID
    
    def __post_init__(self):
        """初始化后处理，设置默认路径"""
        import os
        if self.database_path is None:
            self.database_path = os.path.join(self.workspace_dir, "database.db")
        if self.export_dir is None:
            self.export_dir = os.path.join(self.workspace_dir, "exports")