"""
特征匹配模块

使用FAISS进行GPU加速的近似最近邻匹配
"""
import numpy as np
import faiss
import logging
from typing import List, Dict, Tuple, Set, Optional
import math

from config import FeatureConfig
from core.data_models import MatchData
from storage.database import Database


class FAISSMatcher:
    """基于FAISS的特征匹配器
    
    使用GPU加速的近似最近邻搜索进行特征匹配
    """
    
    def __init__(self, db: Database, config: FeatureConfig):
        """初始化特征匹配器
        
        Args:
            db: 数据库实例
            config: 特征提取配置
        """
        self.db = db
        self.config = config
        self.logger = logging.getLogger("FAISSMatcher")
        
        # 检查FAISS GPU可用性
        self.use_gpu = hasattr(faiss, "StandardGpuResources")
        if not self.use_gpu:
            self.logger.warning("FAISS GPU不可用，将使用CPU进行特征匹配，这会显著降低性能")
        else:
            self.gpu_res = faiss.StandardGpuResources()
            self.logger.info(f"初始化FAISS GPU匹配器，GPU ID: {config.faiss_gpu_id}")
    
    def match_images(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """匹配多张图像
        
        Args:
            image_ids: 图像ID列表
            
        Returns:
            匹配的图像对列表 [(id1, id2), ...]，保证id1 < id2
        """
        self.logger.info(f"开始匹配 {len(image_ids)} 张图像")
        
        # 1. 确定要匹配的图像对
        image_pairs = self._select_image_pairs(image_ids)
        self.logger.info(f"选择了 {len(image_pairs)} 对图像进行匹配")
        
        # 2. 对每对图像进行特征匹配
        matched_pairs = []
        for i, (id1, id2) in enumerate(image_pairs):
            try:
                match_data = self._match_image_pair(id1, id2)
                if match_data is not None and len(match_data.matches) > 0:
                    # 保存匹配结果
                    self.db.save_matches(match_data)
                    matched_pairs.append((id1, id2))
                
                # 打印进度
                if (i + 1) % 100 == 0 or (i + 1) == len(image_pairs):
                    self.logger.info(f"已处理 {i + 1}/{len(image_pairs)} 对图像")
            except Exception as e:
                self.logger.error(f"匹配图像对 ({id1}, {id2}) 时出错: {str(e)}")
        
        self.logger.info(f"成功匹配 {len(matched_pairs)}/{len(image_pairs)} 对图像")
        return matched_pairs
    
    
    def _select_image_pairs(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """选择要匹配的图像对
        
        基于GPS位置或穷举所有可能的对
        
        Args:
            image_ids: 图像ID列表
            
        Returns:
            要匹配的图像对列表 [(id1, id2), ...]，保证id1 < id2
        """
        if self.config.use_gps_filtering:
            return self._select_pairs_by_gps(image_ids)
        else:
            return self._select_all_pairs(image_ids)
    

    def _select_all_pairs(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """选择所有可能的图像对
        
        Args:
            image_ids: 图像ID列表
            
        Returns:
            所有可能的图像对列表 [(id1, id2), ...]，保证id1 < id2
        """
        pairs = []
        for i in range(len(image_ids)):
            for j in range(i + 1, len(image_ids)):
                id1, id2 = image_ids[i], image_ids[j]
                if id1 > id2:
                    id1, id2 = id2, id1
                pairs.append((id1, id2))
        return pairs
    
    def _select_pairs_by_gps(self, image_ids: List[int]) -> List[Tuple[int, int]]:
        """基于GPS位置选择图像对
        
        Args:
            image_ids: 图像ID列表
            
        Returns:
            基于GPS距离筛选的图像对列表 [(id1, id2), ...]，保证id1 < id2
        """
        # 加载所有图像的元数据
        images_with_gps = []
        images_without_gps = []
        
        for image_id in image_ids:
            try:
                meta = self.db.load_image_meta(image_id)
                if meta.has_gps:
                    images_with_gps.append(meta)
                else:
                    images_without_gps.append(meta)
            except Exception as e:
                self.logger.warning(f"加载图像 {image_id} 元数据失败: {str(e)}")
        
        self.logger.info(f"有GPS信息的图像: {len(images_with_gps)}, 无GPS信息的图像: {len(images_without_gps)}")
        
        # 为每个图像找到最近的N个邻居
        pairs = set()
        
        # 处理有GPS信息的图像
        if images_with_gps:
            # 构建GPS坐标数组
            gps_coords = np.array([[meta.gps_lat, meta.gps_lon] for meta in images_with_gps])
            image_ids_with_gps = [meta.image_id for meta in images_with_gps]
            
            # 对于每个图像，找到最近的N个邻居
            for i, meta in enumerate(images_with_gps):
                # 计算与其他所有图像的距离
                distances = []
                for j, other_meta in enumerate(images_with_gps):
                    if i == j:
                        continue
                    
                    # 计算GPS距离（米）
                    dist = self._haversine_distance(
                        meta.gps_lat, meta.gps_lon, 
                        other_meta.gps_lat, other_meta.gps_lon
                    )
                    
                    if dist <= self.config.max_gps_distance:
                        distances.append((j, dist))
                
                # 按距离排序
                distances.sort(key=lambda x: x[1])
                
                # 选择最近的min_matches_pairs个邻居
                for j, _ in distances[:self.config.min_matches_pairs]:
                    id1, id2 = meta.image_id, images_with_gps[j].image_id
                    if id1 > id2:
                        id1, id2 = id2, id1
                    pairs.add((id1, id2))
        
        # 处理没有GPS信息的图像
        # 将它们与所有其他图像匹配
        for meta1 in images_without_gps:
            # 与有GPS信息的图像匹配
            for meta2 in images_with_gps:
                id1, id2 = meta1.image_id, meta2.image_id
                if id1 > id2:
                    id1, id2 = id2, id1
                pairs.add((id1, id2))
            
            # 与其他没有GPS信息的图像匹配
            for meta2 in images_without_gps:
                if meta1.image_id >= meta2.image_id:
                    continue
                pairs.add((meta1.image_id, meta2.image_id))
        
        return list(pairs)
    

    def _match_image_pair(self, id1: int, id2: int) -> Optional[MatchData]:
        """匹配一对图像的特征
        
        Args:
            id1: 第一个图像ID
            id2: 第二个图像ID
            
        Returns:
            匹配数据，如果匹配失败则返回None
        """
        # 加载特征数据
        try:
            features1 = self.db.load_keypoints(id1, load_desc=True)
            features2 = self.db.load_keypoints(id2, load_desc=True)
        except Exception as e:
            self.logger.warning(f"加载图像对 ({id1}, {id2}) 的特征失败: {str(e)}")
            return None
        
        # 检查特征数量
        if features1.num_keypoints == 0 or features2.num_keypoints == 0:
            return None
        
        # 提取描述符
        desc1 = features1.descriptors.astype(np.float32)
        desc2 = features2.descriptors.astype(np.float32)
        
        try:
            # 使用FAISS进行最近邻搜索
            matches12, scores12 = self._find_nearest_neighbors(desc1, desc2)
            
            if len(matches12) == 0:
                self.logger.debug(f"图像对 ({id1}, {id2}) 没有找到匹配")
                return None
            
            # 如果需要双向匹配
            if self.config.mutual_best_match:
                matches21, scores21 = self._find_nearest_neighbors(desc2, desc1)
                
                if len(matches21) == 0:
                    self.logger.debug(f"图像对 ({id1}, {id2}) 反向匹配没有找到匹配")
                    return None
                
                # 找到互为最佳匹配的点
                mutual_matches = []
                mutual_scores = []
                
                for i, (j, score) in enumerate(zip(matches12, scores12)):
                    # 安全检查：确保索引在有效范围内
                    if j < len(matches21):
                        # 检查是否互为最佳匹配
                        if matches21[j] == i:
                            mutual_matches.append((i, j))
                            mutual_scores.append(score)
                
                if not mutual_matches:
                    self.logger.debug(f"图像对 ({id1}, {id2}) 没有互为最佳的匹配")
                    return None
                
                matches = np.array(mutual_matches, dtype=np.int32)
                scores = np.array(mutual_scores, dtype=np.float32)
            else:
                # 单向匹配
                matches = np.column_stack((np.arange(len(matches12)), matches12))
                scores = scores12
            
            # 限制匹配数量
            if len(matches) > self.config.max_num_matches:
                # 按分数排序
                idx = np.argsort(scores)[:self.config.max_num_matches]
                matches = matches[idx]
                scores = scores[idx]
            
            # 创建匹配数据
            return MatchData(
                img_id1=id1,
                img_id2=id2,
                matches=matches,
                scores=scores
            )
        
        except Exception as e:
            self.logger.error(f"匹配图像对 ({id1}, {id2}) 时出错: {str(e)}")
            return None
    

    def _find_nearest_neighbors(self, desc1: np.ndarray, desc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用FAISS查找最近邻
        
        Args:
            desc1: 查询描述符 [N, 128]
            desc2: 参考描述符 [M, 128]
            
        Returns:
            (matches, scores) 元组
            matches: [N'] 数组，表示desc1中每个描述符在desc2中的最近邻索引
            scores: [N'] 数组，表示最近邻的距离比率（第一近邻/第二近邻）
        """
        # 确保描述符是float32类型
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
        
        # 检查描述符数量
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        # 创建FAISS索引
        d = desc2.shape[1]  # 描述符维度
        
        try:
            if self.use_gpu:
                # 使用GPU索引
                index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, min(self.config.faiss_nlist, desc2.shape[0]))
                index = faiss.index_cpu_to_gpu(self.gpu_res, self.config.faiss_gpu_id, index)
            else:
                # 使用CPU索引
                index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, min(self.config.faiss_nlist, desc2.shape[0]))
            
            # 训练索引
            if not index.is_trained:
                index.train(desc2)
            
            # 添加参考描述符
            index.add(desc2)
            
            # 设置搜索参数
            index.nprobe = min(self.config.faiss_nprobe, desc2.shape[0])
            
            # 查找2个最近邻（用于Lowe's比率测试）
            k = min(2, desc2.shape[0])  # 确保k不大于desc2的大小
            distances, indices = index.search(desc1, k)
            
            # 处理只找到1个最近邻的情况
            if k == 1:
                # 只有一个最近邻时，无法应用比率测试，直接返回所有匹配
                matches = indices[:, 0]
                scores = distances[:, 0]
                
                # 过滤无效匹配（-1表示没有找到匹配）
                valid_idx = np.where(matches != -1)[0]
                return matches[valid_idx], scores[valid_idx]
            
            # 应用Lowe's比率测试
            mask = distances[:, 0] < self.config.lowes_ratio * distances[:, 1]
            
            # 计算比率作为分数
            scores = np.ones(len(desc1), dtype=np.float32)
            valid_indices = np.where(distances[:, 1] > 0)[0]
            scores[valid_indices] = distances[valid_indices, 0] / distances[valid_indices, 1]
            
            # 获取最近邻索引和分数
            matches = indices[:, 0]
            
            # 应用比率测试过滤
            matches[~mask] = -1
            
            # 只保留有效匹配
            valid_idx = np.where(matches != -1)[0]
            if len(valid_idx) == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
            
            # 确保所有索引都在有效范围内
            valid_matches = matches[valid_idx]
            valid_scores = scores[valid_idx]
            
            # 再次检查索引是否在有效范围内
            final_valid_idx = np.where(valid_matches < desc2.shape[0])[0]
            return valid_matches[final_valid_idx], valid_scores[final_valid_idx]
            
        except Exception as e:
            self.logger.error(f"FAISS搜索失败: {str(e)}")
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两个GPS坐标之间的距离（米）
        
        使用Haversine公式计算球面距离
        
        Args:
            lat1: 第一个点的纬度（度）
            lon1: 第一个点的经度（度）
            lat2: 第二个点的纬度（度）
            lon2: 第二个点的经度（度）
            
        Returns:
            两点之间的距离（米）
        """
        # 地球半径（米）
        R = 6371000.0
        
        # 转换为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine公式
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance