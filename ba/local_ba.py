"""
局部集束调整模块

实现滑动窗口局部BA，优化最新注册帧及其共视图像
"""
import os
import cv2
import time
import pyceres
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple

from config import BAConfig
from core.scene import SceneManager, View, Point3D
from core.scene_graph import SceneGraph
from ba.base_ba import BundleAdjusterBase
from ba.cost_functions import ReprojectionCost


class LocalBundleAdjuster(BundleAdjusterBase):
    """局部集束调整器
    
    实现滑动窗口局部BA，优化最新注册帧及其共视图像
    """
    
    def __init__(self, config: BAConfig, db=None):
        """初始化局部集束调整器
        
        Args:
            config: BA配置
        """
        super().__init__(config)
        self.window_size = config.local_ba_window_size
        self.fixed_frames = config.local_ba_fixed_frames
        self.db = db


    def optimize(self, scene: SceneManager, **kwargs) -> Dict[str, Any]:
        """执行局部集束调整优化

        Args:
            scene: 场景管理器
            **kwargs: 额外参数，可以包含：
                - target_image_id: 优化目标图像ID，默认为最新注册的图像
                - window_size: 窗口大小，覆盖配置中的值
                - fixed_frames: 固定帧数量，覆盖配置中的值
            
        Returns:
            包含优化统计信息的字典
        """
        start_time = time.time()

        # ── 1. 解析参数 ──────────────────────────────────────────────────────────
        target_image_id = kwargs.get('target_image_id', None)
        window_size = kwargs.get('window_size', self.window_size)
        fixed_frames = kwargs.get('fixed_frames', self.fixed_frames)

        # ── 2. 确定目标图像 ───────────────────────────────────────────────────────
        if target_image_id is None:
            registered_images = scene.get_registered_images()
            self.logger.info(f"已注册图像数量: {len(registered_images)}, IDs: {registered_images}")
            if not registered_images:
                self.logger.warning("没有已注册的图像，无法执行局部BA")
                return {"success": False, "error": "没有已注册的图像"}
            target_image_id = registered_images[-1]

        # 确保目标图像在scene.views中
        if target_image_id not in scene.views:
            self.logger.error(f"目标图像ID {target_image_id} 在scene.views中不存在")
            return {"success": False, "error": f"目标图像ID {target_image_id} 不存在"}
        
        # 确保目标图像已注册
        if not scene.views[target_image_id].is_registered:
            self.logger.error(f"目标图像ID {target_image_id} 未注册")
            return {"success": False, "error": f"目标图像ID {target_image_id} 未注册"}

        # ── 3. 选择局部BA窗口 ─────────────────────────────────────────────────────
        scene_graph   = SceneGraph(scene)
        window_images = scene_graph.find_local_bundle_window(target_image_id, window_size)
        self.logger.info(f"选择的窗口图像: {window_images}")

        if len(window_images) < 2:
            self.logger.warning(f"局部BA窗口中的图像数量不足: {len(window_images)}")
            return {"success": False, "error": "窗口中的图像数量不足"}


        # 转为 set 加速后续 O(1) 查找
        window_image_set = set(window_images)

        # 固定帧也转为 set
        fixed_images = set(scene_graph.find_fixed_images(window_images, fixed_frames))

        self.logger.info(f"执行局部BA，窗口大小: {len(window_images)}，固定帧: {len(fixed_images)}")

        # ── 4. 收集窗口内的相机视图 ───────────────────────────────────────────────
        views_to_optimize: Dict[int, Any] = {
            image_id: scene.views[image_id]
            for image_id in window_images
            if image_id in scene.views
        }
        self.logger.info(f"待优化的视图: {list(views_to_optimize.keys())}")


        # ── 5. 收集相关3D点，区分自由点和锚点 ────────────────────────────────────
        points_to_optimize: Dict[int, Any] = {}
        constant_points: set = set()

        for point_id, point in scene.points3d.items():
            if point.is_outlier:
                continue
            
            observed_by_new_image = False   # 被新加入的图像（图像1）观测到
            observed_by_fixed     = False   # 被固定相机（图像0/2）观测到
            
            for image_id in point.observations:
                if image_id in window_image_set:
                    if image_id in fixed_images:
                        observed_by_fixed = True
                    else:
                        observed_by_new_image = True  # 新图像观测到了
            
            if observed_by_fixed or observed_by_new_image:
                points_to_optimize[point_id] = point
                # 只被固定相机观测、新图像没看到 → 锚点
                if observed_by_fixed and not observed_by_new_image:
                    constant_points.add(point_id)


        self.logger.info(f"优化 {len(views_to_optimize)} 个相机, {len(points_to_optimize)} 个点 "f"(其中 {len(constant_points)} 个点为锚点)")

        # ── 6. 预加载所有窗口内图像的特征点（避免循环内重复IO）────────────────────
        keypoints_cache: Dict[int, Any] = {}
        for image_id in window_image_set:
            try:
                keypoints_cache[image_id] = self.db.load_keypoints(image_id, load_desc=False)
            except Exception as e:
                self.logger.warning(f"预加载图像 {image_id} 特征点失败: {e}")

        # ── 7. 构建 Ceres 问题 ────────────────────────────────────────────────────
        problem = pyceres.Problem()

        # 相机参数块：[rx, ry, rz, tx, ty, tz]，共6个参数
        # 注意：必须持有 numpy array 引用，防止被 GC 回收
        camera_params_map: Dict[int, np.ndarray] = {}

        for image_id, view in views_to_optimize.items():
            rvec, _ = cv2.Rodrigues(view.R)
            params  = np.hstack([rvec.flatten(), view.t.flatten()]).astype(np.float64)
            camera_params_map[image_id] = params
            problem.add_parameter_block(params, 6)
            if image_id in fixed_images:
                problem.set_parameter_block_constant(params)
            
        # 3D点参数块：[X, Y, Z]，共3个参数
        point_params_map: Dict[int, np.ndarray] = {}

        for point_id, point in points_to_optimize.items():
            params = point.xyz.copy().astype(np.float64)
            point_params_map[point_id] = params
            problem.add_parameter_block(params, 3)
            if point_id in constant_points:
                problem.set_parameter_block_constant(params)

        # ── 8. 添加重投影误差残差块 ───────────────────────────────────────────────────
        num_residuals = 0
        num_skipped   = 0
        loss_function = pyceres.HuberLoss(self.config.huber_loss_parameter)
        for point_id, point in points_to_optimize.items():
            point_params = point_params_map[point_id]
            
            for image_id, keypoint_idx in point.observations.items():
                if image_id not in window_image_set:
                    continue

                # 从缓存获取特征点，避免重复IO
                if image_id not in keypoints_cache:
                    num_skipped += 1
                    continue
                
                features = keypoints_cache[image_id]
                if keypoint_idx >= len(features.keypoints_xy):
                    self.logger.warning(
                        f"点 {point_id} 在图像 {image_id} 中的特征点索引越界: "
                        f"{keypoint_idx} >= {len(features.keypoints_xy)}"
                    )
                    num_skipped += 1
                    continue
                
                keypoint      = features.keypoints_xy[keypoint_idx]
                view          = views_to_optimize[image_id]
                camera_params = camera_params_map[image_id]
                
                cost_function = ReprojectionCost(keypoint, view.K)
                
                problem.add_residual_block(
                    cost_function,
                    loss_function,
                    [camera_params, point_params]
                )

                num_residuals += 2  # x 和 y 方向各一个残差


        if num_skipped > 0:
            self.logger.warning(f"跳过了 {num_skipped} 个无效观测")

        # 残差块为空时直接返回，避免 Ceres 报错
        if num_residuals == 0:
            self.logger.warning("没有有效的残差块，跳过BA优化")
            return {"success": False, "error": "没有有效的残差块"}
        

        # ── 9. 配置并执行求解器 ───────────────────────────────────────────────────
        options = pyceres.SolverOptions()
        options.linear_solver_type = self.config.linear_solver_type
        options.max_num_iterations = self.config.max_iterations
        options.function_tolerance = self.config.function_tolerance
        options.gradient_tolerance = self.config.gradient_tolerance
        options.parameter_tolerance = self.config.parameter_tolerance
        options.minimizer_progress_to_stdout = True
        options.num_threads = -1

        # 更新相机姿态      
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        # 打印求解结果
        self.logger.info(summary.BriefReport())

        # 检查是否成功
        if not summary.IsSolutionUsable():
            self.logger.error(f"BA求解失败: {summary.BriefReport()}")
            return False

        # 检查求解是否成功（CONVERGENCE / USER_SUCCESS 均视为成功）
        solver_success = summary.termination_type in (
            pyceres.TerminationType.CONVERGENCE,
            pyceres.TerminationType.USER_SUCCESS,
        )
        if not solver_success:
            self.logger.warning(
                f"BA求解未收敛，终止原因: {summary.termination_type}, "
                f"消息: {summary.message}"
            )

        # ── 10. 回写优化结果到场景 ────────────────────────────────────────────────
        # 直接从 numpy array 读取（pyceres 原地修改参数块内存）
        # 无需调用 get_parameter_block_data，camera_params_map 中的数组已是最新值

        for image_id, params in camera_params_map.items():
            if image_id in fixed_images:
                continue
            
            view = views_to_optimize[image_id]

            rvec = params[:3].reshape(3, 1)
            tvec = params[3:].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            
            view   = views_to_optimize[image_id]
            view.R = R
            view.t = tvec

        for point_id, params in point_params_map.items():
            if point_id in constant_points:
                continue
            points_to_optimize[point_id].xyz = params.copy().astype(np.float32)


        # ── 11. 统计并返回 ────────────────────────────────────────────────────────
        elapsed_time = time.time() - start_time

        stats = {
            "success":          solver_success,
            "num_cameras":      len(views_to_optimize),
            "num_points":       len(points_to_optimize),
            "num_residuals":    num_residuals,
            "initial_cost":     summary.initial_cost,
            "final_cost":       summary.final_cost,
            "termination_type": str(summary.termination_type),
            "elapsed_time":     elapsed_time,
        }

        self.logger.info(
            f"局部BA完成，耗时 {elapsed_time:.2f}s | "
            f"代价: {summary.initial_cost:.4e} → {summary.final_cost:.4e} | "
            f"收敛: {solver_success}"
        )

        return stats