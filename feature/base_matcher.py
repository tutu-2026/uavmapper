"""
特征匹配模块基类

定义特征匹配器的通用接口和流程，子类只需实现核心匹配算法
"""
import math
import logging
import numpy as np
import multiprocessing
from functools import partial
from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Optional

from config import FeatureConfig
from storage.database import Database
from core.data_models import MatchData


class BaseMatcher(ABC):
    """特征匹配器基类

    封装了图像对选择、匹配流程编排、数据库交互等通用逻辑。
    子类只需实现 `_match_descriptors` 方法，提供具体的描述符匹配算法。

    子类实现示例：
        class MyMatcher(BaseMatcher):
            def _init_backend(self):
                # 初始化匹配后端资源
                pass
                
            def _match_descriptors(self, desc1, desc2):
                # 实现具体的匹配算法
                ...
    """

    def __init__(self, db: Database, config: FeatureConfig):
        """初始化特征匹配器基类

        Args:
            db: 数据库实例
            config: 特征提取配置
        """
        self.db = db
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)


    @staticmethod
    def _init_worker():
        """初始化工作进程的资源
    
        这个方法会在每个工作进程启动时被调用。
        子类可以重写此方法以加载模型或初始化其他资源。
        """
        pass


    @abstractmethod
    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """查找最近邻描述符（核心匹配算法，由子类实现）

        Args:
            desc1: 查询描述符，形状 [N, D]，dtype=float32
            desc2: 参考描述符，形状 [M, D]，dtype=float32

        Returns:
            matches: 形状 [K]，int32，desc1中每个有效描述符在desc2中的最近邻索引
            scores: 形状 [K]，float32，对应的匹配得分（越小越好，如距离比率）
        """
        pass


    def match_images(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """匹配多张图像

        Args:
            image_ids: 图像ID列表

        Returns:
            成功匹配的图像对列表 [(id1, id2), ...]，保证 id1 < id2
        """
        self.logger.info(f"开始匹配 {len(image_ids)} 张图像")

        image_pairs = self._select_image_pairs(image_ids)
        self.logger.info(f"选择了 {len(image_pairs)} 对图像进行匹配")

        # 确定进程数
        num_processes = min(self.config.batch_size, multiprocessing.cpu_count())
        self.logger.info(f"使用 {num_processes} 个进程进行并行匹配")        

        # 使用进程池并行处理
        matched_pairs: List[Tuple[int, int]] = []
        self._init_worker()
        for pair in image_pairs:
            match_data = self._match_image_pair(pair[0], pair[1])
            # 保存匹配结果
            self.db.save_matches(match_data)
            matched_pairs.append((pair[0], pair[1]))

        # with multiprocessing.Pool(processes=num_processes, initializer=self._init_worker) as pool:
        #     # 分批处理，以便显示进度
        #     batch_size = max(100, len(image_pairs) // 20) # 每批约5%的任务
        #     for i in range(0, len(image_pairs), batch_size):
        #         batch = image_pairs[i:i+batch_size]
        #         # 准备参数 [(id1, id2), ...] -> [(id1, id2), ...]
        #         args = [(pair[0], pair[1]) for pair in batch]
        #         results = pool.starmap(self._match_image_pair, args)

        #         # 收集成功匹配的结果
        #         for pair, match_data in zip(batch, results):
        #             if match_data is not None:
        #                 self.db.save_matches(match_data)
        #                 matched_pairs.append(pair)
                
        #         # 显示进度
        #         self.logger.info(f"已处理 {min(i+batch_size, len(image_pairs))}/{len(image_pairs)} 对图像")

        self.logger.info(f"成功匹配 {len(matched_pairs)}/{len(image_pairs)} 对图像")
        return matched_pairs


    def _select_image_pairs(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """根据配置选择要匹配的图像对

        Args:
            image_ids: 图像ID列表

        Returns:
            图像对列表 [(id1, id2), ...]，保证 id1 < id2
        """
        if self.config.use_gps_filtering:
            return self._select_pairs_by_gps(image_ids)
        return self._select_all_pairs(image_ids)


    def _select_all_pairs(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """穷举所有可能的图像对

        Args:
            image_ids: 图像ID列表

        Returns:
            所有图像对 [(id1, id2), ...]，保证 id1 < id2
        """
        pairs: List[Tuple[int, int]] = []
        n = len(image_ids)
        for i in range(n):
            for j in range(i + 1, n):
                id1, id2 = image_ids[i], image_ids[j]
                pairs.append((min(id1, id2), max(id1, id2)))
        return pairs


    def _select_pairs_by_gps(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """基于GPS位置筛选图像对

        有GPS信息的图像：按Haversine距离选取最近的N个邻居。
        无GPS信息的图像：与所有其他图像配对（保底策略）。

        Args:
            image_ids: 图像ID列表

        Returns:
            筛选后的图像对 [(id1, id2), ...]，保证 id1 < id2
        """
        images_with_gps = []
        images_without_gps = []

        # 区分有GPS和无GPS的图像
        for image_id in image_ids:
            try:
                meta = self.db.load_image_meta(image_id)
                if meta.has_gps:
                    images_with_gps.append(meta)
                else:
                    images_without_gps.append(meta)
            except Exception as e:
                self.logger.warning(f"加载图像 {image_id} 元数据失败: {e}")

        self.logger.info(
            f"有GPS信息的图像: {len(images_with_gps)}，"
            f"无GPS信息的图像: {len(images_without_gps)}"
        )

        pairs: Set[Tuple[int, int]] = set()

        # 有GPS的图像：按距离选邻居
        for i, meta in enumerate(images_with_gps):
            neighbors = []
            for j, other in enumerate(images_with_gps):
                if i == j:
                    continue
                    
                # 计算GPS距离
                dist = self._haversine_distance(
                    meta.gps_lat, meta.gps_lon,
                    other.gps_lat, other.gps_lon,
                )
                
                # 筛选距离范围内的图像
                if dist <= self.config.max_gps_distance:
                    neighbors.append((j, dist))

            # 按距离排序并选择最近的N个邻居
            neighbors.sort(key=lambda x: x[1])
            for j, _ in neighbors[:self.config.min_matches_pairs]:
                id1, id2 = meta.image_id, images_with_gps[j].image_id
                pairs.add((min(id1, id2), max(id1, id2)))

        # 无GPS的图像：与所有图像配对
        for meta1 in images_without_gps:
            # 与有GPS的图像配对
            for meta2 in images_with_gps:
                id1, id2 = meta1.image_id, meta2.image_id
                pairs.add((min(id1, id2), max(id1, id2)))
                
            # 与其他无GPS的图像配对
            for meta2 in images_without_gps:
                if meta1.image_id < meta2.image_id:
                    pairs.add((meta1.image_id, meta2.image_id))

        return list(pairs)


    def _match_image_pair(self, id1: int, id2: int) -> Optional[MatchData]:
        """匹配一对图像的特征（模板方法）

        流程：加载特征 → 调用_match_descriptors → 互匹配过滤 → 数量限制

        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID

        Returns:
            MatchData，若匹配失败则返回None
        """
        # 1. 加载特征
        try:
            features1 = self.db.load_keypoints(id1, load_desc=True)
            features2 = self.db.load_keypoints(id2, load_desc=True)
        except Exception as e:
            self.logger.warning(f"加载图像对 ({id1}, {id2}) 的特征失败: {e}")
            return None

        # 检查特征点数量
        if features1.num_keypoints == 0 or features2.num_keypoints == 0:
            return None

        # 转换描述符类型
        desc1 = features1.descriptors.astype(np.float32)
        desc2 = features2.descriptors.astype(np.float32)

        try:
            # 2. 正向匹配
            matches12, scores12 = self._match_descriptors(desc1, desc2)
            if len(matches12) == 0:
                self.logger.debug(f"图像对 ({id1}, {id2}) 正向匹配无结果")
                return None

            # # 3. 互匹配过滤（可选）
            # if self.config.mutual_best_match:
            #     matches, scores = self._apply_mutual_filter(matches12, scores12, desc2, desc1)
            # else:
            #     # 不使用互匹配时，构建匹配对
            #     # matches = np.column_stack((np.arange(len(matches12), dtype=np.int64), matches12))
            #     scores = scores12

            if len(matches12) == 0:
                self.logger.debug(f"图像对 ({id1}, {id2}) 互匹配过滤后无结果")
                return None

            # 4. 限制最大匹配数量（保留得分最低的匹配）
            if len(matches12) > self.config.max_num_matches:
                keep = np.argsort(scores12)[:self.config.max_num_matches]
                matches12 = matches12[keep]
                scores12 = scores12[keep]

            return MatchData(img_id1=id1, img_id2=id2, matches=matches12, scores=scores12)

        except Exception as e:
            self.logger.error(f"匹配图像对 ({id1}, {id2}) 时出错: {e}")
            return None


    def _apply_mutual_filter(
        self,
        matches12: np.ndarray,
        scores12: np.ndarray,
        desc2: np.ndarray,
        desc1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """互匹配过滤：保留双向互为最佳匹配的点对

        Args:
            matches12: 正向匹配索引 [N]
            scores12: 正向匹配得分 [N]
            desc2: 参考描述符（用于反向匹配）
            desc1: 查询描述符（用于反向匹配）

        Returns:
            (matches, scores)，matches形状[K, 2]，scores形状[K]
        """
        # 执行反向匹配
        matches21, _ = self._match_descriptors(desc2, desc1)
        if len(matches21) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

        # 收集互为最佳匹配的点对
        mutual_matches = []
        mutual_scores = []
        for i, (j, score) in enumerate(zip(matches12, scores12)):
            if j < len(matches21) and matches21[j] == i:
                mutual_matches.append((i, int(j)))
                mutual_scores.append(score)

        if not mutual_matches:
            return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

        return (
            np.array(mutual_matches, dtype=np.int32),
            np.array(mutual_scores, dtype=np.float32),
        )


    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """计算两个GPS坐标之间的球面距离（Haversine公式）

        Args:
            lat1, lon1: 第一个点的纬度/经度（度）
            lat2, lon2: 第二个点的纬度/经度（度）

        Returns:
            距离（米）
        """
        R = 6_371_000.0  # 地球平均半径（米）
        
        # 转换为弧度
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
        
        # 计算差值
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        # Haversine公式
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))