"""
全局集束调整模块

实现全局BA，优化所有相机和3D点，可选添加GPS先验约束
"""
import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import time

import pyceres

from config import BAConfig
from core.scene import SceneManager, View, Point3D
from ba.base_ba import BundleAdjusterBase
from ba.cost_functions import ReprojectionCost, GPSPositionCost


class GlobalBundleAdjuster(BundleAdjusterBase):
    """全局集束调整器
    
    实现全局BA，优化所有相机和3D点，可选添加GPS先验约束
    """
    
    def __init__(self, config: BAConfig):
        """初始化全局集束调整器
        
        Args:
            config: BA配置
        """
        super().__init__(config)
        self.use_gps_prior = config.use_gps_prior
        self.gps_prior_weight = config.gps_prior_weight
    
    def optimize(self, scene: SceneManager, **kwargs) -> Dict[str, Any]:
        """执行全局集束调整优化
        
        Args:
            scene: 场景管理器
            **kwargs: 额外参数，可以包含：
                - fixed_cameras: 要固定的相机ID列表
                - max_iterations: 最大迭代次数，覆盖配置中的值
            
        Returns:
            包含优化统计信息的字典
        """
        start_time = time.time()
        
        # 获取参数
        fixed_cameras = kwargs.get('fixed_cameras', [])
        max_iterations = kwargs.get('max_iterations', self.config.max_iterations)
        
        # 获取所有已注册的相机
        registered_images = scene.get_registered_images()
        if not registered_images:
            self.logger.warning("没有已注册的图像，无法执行全局BA")
            return {"success": False, "error": "没有已注册的图像"}
        
        # 如果没有指定固定相机，默认固定第一帧
        if not fixed_cameras and registered_images:
            fixed_cameras = [registered_images[0]]
        
        self.logger.info(f"执行全局BA，优化 {len(registered_images)} 个相机，固定 {len(fixed_cameras)} 个相机")
        
        # 收集所有相机和有效的3D点
        cameras_to_optimize = {}
        points_to_optimize = {}
        
        # 收集所有已注册的相机
        for image_id in registered_images:
            if image_id in scene.cameras:
                cameras_to_optimize[image_id] = scene.cameras[image_id]
        
        # 收集所有非异常点
        for point_id, point in scene.points3d.items():
            if not point.is_outlier:
                points_to_optimize[point_id] = point
        
        self.logger.info(f"优化 {len(cameras_to_optimize)} 个相机, {len(points_to_optimize)} 个点")
        
        # 创建Ceres问题
        problem = pyceres.Problem()
        
        # 相机参数块：旋转(3) + 平移(3) = 6个参数
        camera_blocks = {}
        
        for image_id, camera in cameras_to_optimize.items():
            # 将旋转矩阵转换为旋转向量
            rvec, _ = cv2.Rodrigues(camera.R)
            
            # 创建参数块：旋转向量(3) + 平移向量(3)
            camera_params = np.hstack([rvec.flatten(), camera.t.flatten()]).astype(np.float64)
            
            # 添加参数块
            block_id = problem.add_parameter_block(camera_params)
            camera_blocks[image_id] = block_id
            
            # 如果是固定相机，则设置为常量
            if image_id in fixed_cameras:
                problem.set_parameter_block_constant(block_id)
            
            # 如果启用GPS先验且相机有GPS数据
            if self.use_gps_prior and camera.gps_local_xyz is not None:
                # 添加GPS位置先验约束
                gps_cost = GPSPositionCost(camera.gps_local_xyz, self.gps_prior_weight)
                problem.add_residual_block(
                    gps_cost,
                    pyceres.LossFunction("HUBER", 1.0),
                    block_id
                )
        
        # 3D点参数块：每个点3个参数(X,Y,Z)
        point_blocks = {}
        
        for point_id, point in points_to_optimize.items():
            # 创建参数块：3D点坐标
            point_params = point.xyz.copy().astype(np.float64)
            
            # 添加参数块
            block_id = problem.add_parameter_block(point_params)
            point_blocks[point_id] = block_id
        
        # 添加重投影误差项
        num_residuals = 0
        
        for point_id, point in points_to_optimize.items():
            point_block = point_blocks[point_id]
            
            for image_id, keypoint_idx in point.observations.items():
                if image_id not in camera_blocks:
                    continue  # 跳过未注册的相机
                
                camera = cameras_to_optimize[image_id]
                camera_block = camera_blocks[image_id]
                
                # 加载特征点坐标
                try:
                    features = scene._db.load_keypoints(image_id, load_desc=False)
                    keypoint = features.keypoints_xy[keypoint_idx]
                except Exception as e:
                    self.logger.warning(f"加载图像 {image_id} 的特征点失败: {str(e)}")
                    continue
                
                # 创建重投影误差项
                cost_function = ReprojectionCost(keypoint, camera.K)
                
                # 添加残差块
                problem.add_residual_block(
                    cost_function,
                    pyceres.LossFunction(self.config.loss_function_type, self.config.huber_loss_parameter),
                    camera_block,
                    point_block
                )
                
                num_residuals += 2  # x和y方向各一个残差
        
        # 配置求解器选项
        options = pyceres.SolverOptions()
        options.linear_solver_type = self.config.linear_solver_type
        options.max_num_iterations = max_iterations
        options.function_tolerance = self.config.function_tolerance
        options.gradient_tolerance = self.config.gradient_tolerance
        options.parameter_tolerance = self.config.parameter_tolerance
        options.minimizer_progress_to_stdout = False
        
        # 求解优化问题
        summary = pyceres.solve(options, problem)
        
        # 更新相机参数
        for image_id, block_id in camera_blocks.items():
            if image_id in fixed_cameras:
                continue  # 跳过固定相机
                
            camera = cameras_to_optimize[image_id]
            camera_params = problem.get_parameter_block_data(block_id)
            
            # 分离旋转向量和平移向量
            rvec = camera_params[:3].reshape(3, 1)
            tvec = camera_params[3:].reshape(3, 1)
            
            # 将旋转向量转换为旋转矩阵
            R, _ = cv2.Rodrigues(rvec)
            
            # 更新相机参数
            camera.R = R
            camera.t = tvec
        
        # 更新3D点坐标
        for point_id, block_id in point_blocks.items():
            point = points_to_optimize[point_id]
            point_params = problem.get_parameter_block_data(block_id)
            
            # 更新点坐标
            point.xyz = point_params
        
        # 计算统计信息
        elapsed_time = time.time() - start_time
        
        stats = {
            "success": True,
            "num_cameras": len(cameras_to_optimize),
            "num_points": len(points_to_optimize),
            "num_residuals": num_residuals,
            "num_iterations": summary.num_successful_iterations,
            "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost,
            "elapsed_time": elapsed_time
        }
        
        self.logger.info(f"全局BA完成，耗时 {elapsed_time:.2f} 秒，"
                         f"初始代价: {summary.initial_cost:.6e}, "
                         f"最终代价: {summary.final_cost:.6e}, "
                         f"迭代次数: {summary.num_successful_iterations}")
        
        return stats