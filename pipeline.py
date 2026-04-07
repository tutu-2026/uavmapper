"""
主流程编排模块

包含特征提取、SFM重建和结果导出的主要流程
"""
import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple, Any

from config import FeatureConfig, BAConfig, SFMConfig, WorkspaceConfig
from core.data_models import ImageMeta
from storage.workspace import Workspace
from storage.database import Database
from feature.xfeat_extractor import XFeatExtractor
from feature.matcher import FAISSMatcher
from feature.xfeat_matcher import XFeatMatcher
from feature.verifier import PymagsacVerifier
from sfm.initializer import SFMInitializer
from sfm.registrar import IncrementalRegistrar
from sfm.triangulator import Triangulator
from ba.local_ba import LocalBundleAdjuster
from ba.global_ba import GlobalBundleAdjuster
from io_utils.image_loader import ImageLoader
from io_utils.colmap_io import COLMAPWriter
from io_utils.ply_writer import PLYWriter
from core.scene import SceneManager


class Pipeline:
    """SFM重建流水线"""
    
    def __init__(self, config: WorkspaceConfig):
        """初始化重建流水线
        
        Args:
            config: 工作空间配置
        """
        self.config = config
        self.workspace = Workspace(config)
        self.db = self.workspace.get_database()
        self.logger = logging.getLogger("Pipeline")
        
        # 确保导出目录存在
        os.makedirs(config.export_dir, exist_ok=True)
        
        # 默认配置
        self.feature_config = FeatureConfig()
        self.ba_config = BAConfig()
        self.sfm_config = SFMConfig()
        
    def run_feature_extraction(self, feature_config: Optional[FeatureConfig] = None) -> Dict[str, Any]:
        """运行特征提取流程
        
        Args:
            feature_config: 特征提取配置，如果为None则使用默认配置
            
        Returns:
            包含处理统计信息的字典
        """
        self.feature_config = feature_config or self.feature_config
        start_time = time.time()
        
        # 1. 加载图像元数据
        self.logger.info("开始加载图像...")
        image_loader = ImageLoader()
        images = image_loader.load_directory(self.config.image_dir)
        self.logger.info(f"加载了 {len(images)} 张图像")
        
        # # 2. 保存图像元数据到数据库
        # self.logger.info("保存图像元数据到数据库...")
        # for img in images:
        #     self.db.save_camera(img)
        #     self.db.save_image(img)
        
        # # 3. 提取特征
        # self.logger.info("开始提取特征...")
        # extractor = XFeatExtractor(self.feature_config.num_features)
        # # XFeatExtractor._init_worker()
        # # extractor.extract(images[0])
        # num_processes = self.config.num_cpu_threads
        # features_dict = extractor.extract_batch(images, num_processes=num_processes)

        # # 4. 保存提取的特征到数据库
        # self.logger.info("保存提取的特征到数据库...")
        # for image_id, features in features_dict.items():
        #     self.db.save_keypoints(features)
        #     self.db.save_descriptors(image_id, features.descriptors)


        # # 4. 特征匹配
        # self.logger.info("开始特征匹配...")
        # matcher = XFeatMatcher(self.db, self.feature_config)
        # image_ids = [img.image_id for img in images]
        # match_pairs = matcher.match_images(image_ids)
        # self.logger.info(f"生成了 {len(match_pairs)} 对匹配")
        
        # 5. 几何验证
        self.logger.info("开始几何验证...")
        verifier = PymagsacVerifier(self.db, self.feature_config)
        match_pairs = self.db.load_all_pairs()
        valid_pairs = verifier.verify_image_pairs(match_pairs)
        self.logger.info(f"验证通过 {len(valid_pairs)} 对匹配")
        
        elapsed = time.time() - start_time
        stats = {
            "num_images": len(images),
            #"num_match_pairs": len(match_pairs),
            #"num_valid_pairs": len(valid_pairs),
            "elapsed_time": elapsed
        }
        
        self.logger.info(f"特征处理完成，耗时 {elapsed:.2f} 秒")
        return stats
    
    def run_sfm(self, sfm_config: Optional[SFMConfig] = None, ba_config: Optional[BAConfig] = None) -> Dict[str, Any]:
        """运行SFM重建流程
        
        Args:
            sfm_config: SFM配置，如果为None则使用默认配置
            ba_config: BA配置，如果为None则使用默认配置
            
        Returns:
            包含重建统计信息的字典
        """
        self.sfm_config = sfm_config or self.sfm_config
        self.ba_config = ba_config or self.ba_config
        start_time = time.time()
        
        # 1. 初始化场景管理器
        scene = SceneManager()
        
        # 2. 设置BA优化器
        local_ba = LocalBundleAdjuster(self.ba_config, db=self.db)
        global_ba = GlobalBundleAdjuster(self.ba_config) if self.ba_config.use_gps_prior else None

        # 3. 初始化SFM
        self.logger.info("开始SFM初始化...")
        initializer = SFMInitializer(self.db, self.sfm_config)
        init_success, init_image_ids = initializer.initialize(scene, local_ba)
        
        if not init_success:
            self.logger.error("SFM初始化失败")
            return {"success": False, "error": "SFM初始化失败"}
        
        # 4. 设置三角化器
        triangulator = Triangulator(self.sfm_config, self.db)
        
        # 5. 增量式SFM注册
        self.logger.info("开始增量式SFM注册...")
        registrar = IncrementalRegistrar(self.db, scene, triangulator, self.sfm_config)
        
        # 记录统计信息
        registered_count = len(init_image_ids)
        registered_images = set(init_image_ids)
        total_images = len(self.db.load_all_images())
        last_medium_ba = 0
        last_global_ba = 0
        
        while True:
            # 寻找下一个最佳图像进行注册
            next_image_ids = registrar.find_next_best_images()
            
            if not next_image_ids:
                self.logger.info("没有更多可注册的图像，SFM完成")
                break

            image_added = False
            # 注册图像
            for next_image_id in next_image_ids:
                image_added |= registrar.register_next_image(next_image_id)
                if image_added:
                    registered_count += 1
                    registered_images.add(next_image_id)
                    self.logger.info(f"成功注册图像 {next_image_id}, 进度: {registered_count}/{total_images}")
                else:
                    self.logger.warning(f"无法注册图像 {next_image_id}")

            # 触发局部BA
            self._trigger_ba(scene, local_ba, global_ba, registered_count, last_medium_ba, last_global_ba)
            
            # 更新最后BA触发点
            if registered_count % self.ba_config.medium_ba_trigger_freq == 0:
                last_medium_ba = registered_count
            if registered_count % self.ba_config.global_ba_trigger_freq == 0:
                last_global_ba = registered_count


        # 最后进行一次全局BA
        if self.ba_config.use_gps_prior and global_ba is not None:
            self.logger.info("执行最终全局BA...")
            global_ba.optimize(scene)
        
        elapsed = time.time() - start_time
        stats = {
            "success": True,
            "registered_images": len(registered_images),
            "total_images": total_images,
            "registration_ratio": len(registered_images) / total_images if total_images > 0 else 0,
            "points3d_count": len(scene.points3d),
            "elapsed_time": elapsed
        }
        
        self.logger.info(f"SFM重建完成，耗时 {elapsed:.2f} 秒")
        self.logger.info(f"注册了 {stats['registered_images']}/{stats['total_images']} 张图像 "f"({stats['registration_ratio']*100:.1f}%)")
        self.logger.info(f"重建了 {stats['points3d_count']} 个3D点")
        
        # 保存场景到工作区
        self.workspace.save_scene(scene)
        
        return stats
    

    def _trigger_ba(self, scene: SceneManager, 
                    local_ba: LocalBundleAdjuster, 
                    global_ba: Optional[GlobalBundleAdjuster], 
                    current_frame: int,
                    last_medium_ba: int, 
                    last_global_ba: int) -> None:
        """根据配置触发不同级别的BA
        
        Args:
            scene: 场景管理器
            local_ba: 局部BA优化器
            global_ba: 全局BA优化器
            current_frame: 当前已注册的帧数
            last_medium_ba: 上次中等BA的帧数
            last_global_ba: 上次全局BA的帧数
        """
        
        # 每次注册新帧后都进行局部BA
        self.logger.info(f"执行局部BA...")
        local_ba.optimize(scene)
        
        # 每N帧触发中等BA (取消注释以启用)
        if (current_frame - last_medium_ba) >= self.ba_config.medium_ba_trigger_freq:
            self.logger.info(f"执行中等规模BA...")
            # 使用更大窗口的局部BA
            medium_ba = LocalBundleAdjuster(self.ba_config)
            medium_ba.window_size = self.ba_config.medium_ba_window_size
            medium_ba.optimize(scene)
        
        # 每M帧触发全局BA (取消注释以启用)
        # if global_ba is not None and (current_frame - last_global_ba) >= self.ba_config.global_ba_trigger_freq:
        #     self.logger.info(f"执行全局BA...")
        #     global_ba.optimize(scene)
    

    def run_export(self) -> Dict[str, str]:
        """导出重建结果
        
        Returns:
            包含导出文件路径的字典
        """
        self.logger.info("开始导出重建结果...")
        scene = self.workspace.load_scene()
        
        export_paths = {}
        
        # 导出COLMAP格式
        if self.config.export_colmap:
            colmap_dir = os.path.join(self.config.export_dir, "colmap")
            os.makedirs(colmap_dir, exist_ok=True)
            
            writer = COLMAPWriter()
            writer.write(scene, colmap_dir)
            
            export_paths["colmap"] = colmap_dir
            self.logger.info(f"已导出COLMAP格式到 {colmap_dir}")
        
        # 导出PLY点云
        if self.config.export_ply:
            ply_path = os.path.join(self.config.export_dir, "points3d.ply")
            
            ply_writer = PLYWriter()
            ply_writer.write(scene, ply_path)
            
            export_paths["ply"] = ply_path
            self.logger.info(f"已导出PLY点云到 {ply_path}")
        
        return export_paths

    def visualize_feature_matches(
        self, 
        image_id1: int, 
        image_id2: int, 
        output_path: Optional[str] = None
    ) -> None:
        """可视化两张图像之间的特征点匹配
        
        Args:
            image_id1 (int): 第一张图像的 ID。
            image_id2 (int): 第二张图像的 ID。
            output_path (str, optional): 如果提供，将保存结果图像到指定路径。
        """
        self.logger.info(f"开始可视化图像 {image_id1} 和 {image_id2} 的特征点匹配...")

        # 从数据库加载图像元数据
        img_meta1 = self.db.load_image_meta(image_id1)
        img_meta2 = self.db.load_image_meta(image_id2)

        if img_meta1 is None or img_meta2 is None:
            self.logger.error(f"无法加载图像元数据，ID: {image_id1}, {image_id2}")
            return

        # 从数据库加载特征点和匹配对
        keypoints1 = self.db.load_keypoints(image_id1)
        keypoints2 = self.db.load_keypoints(image_id2)
        # matches = self.db.load_matches(image_id1, image_id2)
        matches = self.db.load_inlier_matches(image_id1, image_id2)

        if keypoints1 is None or keypoints2 is None or matches is None:
            self.logger.error(f"无法加载特征点或匹配对，ID: {image_id1}, {image_id2}")
            return

        # 提取图像路径
        img1_path = img_meta1.image_path
        img2_path = img_meta2.image_path

        from vo_utils.visualization import draw_feature_matches
        # 调用绘制工具
        draw_feature_matches(
            img1_path=img1_path,
            img2_path=img2_path,
            keypoints1=keypoints1,
            keypoints2=keypoints2,
            matches=matches,
            output_path=output_path,
        )

        self.logger.info(f"特征点匹配可视化完成，结果保存到 {output_path if output_path else '屏幕显示'}")


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """设置日志配置
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        level: 日志级别
    """
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 创建控制台处理器
    console = logging.StreamHandler()
    console.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='无人机航测三维重建流水线')
    parser.add_argument('--image_dir', required=True, help='图像目录')
    parser.add_argument('--workspace_dir', required=True, help='工作空间目录')
    parser.add_argument('--export_dir', help='导出目录')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--num_threads', type=int, default=8, help='CPU线程数')
    parser.add_argument('--log_file', help='日志文件路径')
    parser.add_argument('--feature_only', action='store_true', help='只运行特征提取')
    parser.add_argument('--sfm_only', action='store_true', help='只运行SFM重建')
    parser.add_argument('--export_only', action='store_true', help='只运行结果导出')
    parser.add_argument('--visualize_matches', nargs=2, metavar=('IMAGE_ID1', 'IMAGE_ID2'), help='可视化两张图像之间的特征点匹配')

    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_file)
    
    # 创建配置
    config = WorkspaceConfig(
        image_dir=args.image_dir,
        workspace_dir=args.workspace_dir,
        export_dir=args.export_dir,
        gpu_id=args.gpu_id,
        num_cpu_threads=args.num_threads
    )
    
    # 创建流水线
    pipeline = Pipeline(config)
    
    # 运行流水线
    if args.export_only:
        pipeline.run_export()
    elif args.sfm_only:
        pipeline.run_sfm()
        pipeline.run_export()
    elif args.feature_only:
        pipeline.run_feature_extraction()
    else:
        # 运行完整流水线
        pipeline.run_feature_extraction()
        pipeline.run_sfm()
        pipeline.run_export()


if __name__ == "__main__":
    # 正常的命令行入口
    if len(sys.argv) > 1:
        main()
    else:
        # 调试模式
        setup_logging(level=logging.INFO)
        
        # 创建测试配置
        config = WorkspaceConfig(
            image_dir=r"E:\uavData\new_test\images",
            workspace_dir=r"E:\uavData\new_test\workspace",
            export_dir=r"E:\uavData\new_test",
            gpu_id=0,
            num_cpu_threads=4
        )
        
        # 创建流水线
        pipeline = Pipeline(config)
        
        # 根据需要调用不同的方法进行调试
        # pipeline.run_feature_extraction()
        # pipeline.visualize_feature_matches(image_id1=0, image_id2=2, output_path="matches_result.jpg")
        pipeline.run_sfm()
        # pipeline.run_export()