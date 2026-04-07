import os
import cv2
import time
import torch
import multiprocessing
from pathlib import Path
from functools import partial

# 全局变量用于存储模型
global_xfeat = None

def init_worker():
    """初始化工作进程，加载模型到全局变量"""
    global global_xfeat
    global_xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, force_reload=False, skip_validation=True)
    print("模型已在工作进程中加载")

def process_single_image(img_path, top_k=8000, scale_factor=0.5):
    # 使用全局模型变量
    global global_xfeat
    
    # print(f"处理图片: {img_path}")
    
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None
    
    # # 记录原始尺寸
    # original_height, original_width = img.shape[:2]
    
    # # 缩放图像
    # new_width = int(original_width * scale_factor)
    # new_height = int(original_height * scale_factor)
    # img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 转换颜色空间
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # print(f"原始尺寸: {original_width}x{original_height}, 缩放后: {new_width}x{new_height}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 提取特征
    output = global_xfeat.detectAndCompute(img, top_k=top_k)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    result = {
        'image': str(img_path),
        'features': len(output['keypoints']),
        'time': process_time,
    }
    
    print(f"特征点数量: {result['features']}")
    print(f"处理时间: {process_time:.4f} 秒")
    print("-" * 50)
    
    return result

def process_images(input_folder, top_k=8000, scale_factor=0.5, num_processes=None):
    # 如果未指定进程数，则使用CPU核心数
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 获取文件夹中的所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
    
    print(f"找到 {len(image_files)} 张图片")
    # print(f"图像缩放比例: {scale_factor}")
    
    # 创建进程池
    start_total = time.time()
    
    # 创建部分函数，固定一些参数
    process_func = partial(process_single_image, top_k=top_k, scale_factor=scale_factor)
    
    # 使用进程池并行处理图像，并在初始化时加载模型
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
        results = pool.map(process_func, image_files)
    
    # 过滤掉None结果（处理失败的图像）
    results = [r for r in results if r is not None]
    
    # 计算总处理时间
    total_time = time.time() - start_total
    
    # 输出总结
    if results:
        processing_time = sum(r['time'] for r in results)
        avg_time = processing_time / len(results)
        avg_features = sum(r['features'] for r in results) / len(results)
        
        print("\n处理总结:")
        print(f"总图片数: {len(results)}")
        print(f"CPU核心数: {multiprocessing.cpu_count()}")
        print(f"使用进程数: {num_processes}")
        print(f"总处理时间: {total_time:.4f} 秒")
        print(f"累计处理时间: {processing_time:.4f} 秒")
        print(f"平均处理时间: {avg_time:.4f} 秒/图")
        print(f"平均特征点数: {avg_features:.1f}")
        print(f"图像缩放比例: {scale_factor}")
        print(f"加速比: {processing_time/total_time:.2f}x")
    
    return results

if __name__ == "__main__":
    # 定义输入文件夹路径
    input_folder = r"F:\uavData\project_69\images"
    
    # 处理图片并提取特征，缩放比例为0.5（缩小为原来的一半）
    # 可以通过num_processes参数指定使用的进程数，默认使用所有可用CPU核心
    results = process_images(input_folder, top_k=8000, scale_factor=0.5, num_processes=4)