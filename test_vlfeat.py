import vlfeat
import numpy as np
from PIL import Image
import os
import time
import glob
import concurrent.futures
from tqdm import tqdm  # 用于显示进度条，如果没有安装可以使用 pip install tqdm

def extract_sift_features(image_path):
    """从单个图像提取SIFT特征"""
    try:
        # 读取图像，转为灰度 float32
        img = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
        
        # 提取 SIFT 特征
        # 返回: frames(4xN) 和 descriptors(128xN)
        start_time = time.time()
        frames, descriptors = vlfeat.vl_sift(img, first_octave=0)
        processing_time = time.time() - start_time
        
        # 注意：在多进程中返回大型数组可能会导致序列化开销
        # 如果只需要统计信息而不需要实际的特征数据，可以只返回必要信息
        return {
            'path': image_path,
            'num_features': frames.shape[1],
            'processing_time': processing_time,
            # 如果不需要实际的特征数据，可以注释掉下面两行
            'frames': frames,
            'descriptors': descriptors
        }
    except Exception as e:
        print(f"处理图像 {os.path.basename(image_path)} 时出错: {str(e)}")
        return None

def process_image_folder(folder_path, max_workers=None, image_extensions=('.jpg', '.jpeg', '.png')):
    """处理文件夹中的所有图像并提取特征，使用多进程并行处理
    
    参数:
        folder_path: 图像文件夹路径
        max_workers: 最大进程数，默认为None（使用系统默认值，通常是CPU核心数）
        image_extensions: 要处理的图像文件扩展名
    """
    # 获取文件夹中所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    
    if not image_files:
        print(f"在文件夹 {folder_path} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    results = []
    total_start_time = time.time()
    
    # 使用进程池并行处理图像
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_image = {executor.submit(extract_sift_features, image_path): image_path for image_path in image_files}
        
        # 使用tqdm创建进度条
        with tqdm(total=len(image_files), desc="处理图像") as pbar:
            for future in concurrent.futures.as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"处理图像 {os.path.basename(image_path)} 时发生异常: {str(e)}")
                finally:
                    pbar.update(1)
    
    total_time = time.time() - total_start_time
    
    # 按处理完成顺序排序结果，便于查看
    results.sort(key=lambda x: os.path.basename(x['path']))
    
    # 输出每个图像的处理结果
    print("\n各图像处理结果:")
    for result in results:
        print(f"图像: {os.path.basename(result['path'])}")
        print(f"  特征点数量: {result['num_features']}")
        print(f"  处理时间: {result['processing_time']:.2f} 秒")
    
    # 输出统计信息
    if results:
        total_features = sum(r['num_features'] for r in results)
        avg_features = total_features / len(results)
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        
        print("\n统计信息:")
        print(f"处理的图像总数: {len(results)}")
        print(f"提取的特征点总数: {total_features}")
        print(f"平均每张图像的特征点数: {avg_features:.2f}")
        print(f"平均每张图像的处理时间: {avg_time:.2f} 秒")
        print(f"总处理时间: {total_time:.2f} 秒")
        print(f"加速比: {sum(r['processing_time'] for r in results) / total_time:.2f}x")
    
    return results

if __name__ == "__main__":
    # 定义要处理的图像文件夹路径
    image_folder = r"E:\uavData\202510161_20251016170442\images"
    
    # 处理文件夹中的所有图像，使用多进程
    # 可以指定进程数，例如 max_workers=4，或者不指定使用系统默认值
    results = process_image_folder(image_folder, max_workers=None)