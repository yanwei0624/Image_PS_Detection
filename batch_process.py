import sys
import cv2
from ForgeryDetection import Detect
import re
from datetime import datetime
import os.path as path
from PIL import Image, ExifTags, ImageChops
import os
import time
from optparse import OptionParser

import double_jpeg_compression
import noise_variance
import copy_move_cfa
import numpy as np

# copy-move parameters for CFA detection
cmd = OptionParser("使用方法: %prog image_file [options]")
cmd.add_option('', '--imauto',
               help='自动搜索相同区域. (默认: %default)', default=1)
cmd.add_option('', '--imblev',
               help='图像细节模糊级别. (默认: %default)', default=3)  # 降低模糊级别
cmd.add_option('', '--impalred',
               help='图像调色板减少因子. (默认: %default)', default=30)  # 增加调色板减少因子
cmd.add_option(
    '', '--rgsim', help='区域相似度阈值. (默认: %default)', default=5)
cmd.add_option(
    '', '--rgsize', help='区域大小阈值. (默认: %default)', default=1.5)
cmd.add_option(
    '', '--blsim', help='块相似度阈值. (默认: %default)', default=200)
cmd.add_option('', '--blcoldev',
               help='块颜色偏差阈值. (默认: %default)', default=0.2)
cmd.add_option(
    '', '--blint', help='块交叉阈值. (默认: %default)', default=0.2)
opt, args = cmd.parse_args()


def PrintBoundary():
    for i in range(50):
        print('*', end='')
    print()


def analyze_image_authenticity(results):
    """
    分析图像真实性
    :param results: 包含各项检测结果的字典
    :return: 真实性分析结论
    """
    # 检测项目计数
    suspicious_count = 0
    total_tests = 0
    
    # 双重JPEG压缩检测
    total_tests += 1
    if results.get('double_compressed') == True:
        suspicious_count += 1
        print("  - 检测到双重JPEG压缩 (可能经过多次保存或编辑)")
    
    # 元数据检测
    total_tests += 1
    if results.get('metadata') == False:
        suspicious_count += 1
        print("  - 未找到图像元数据 (可能经过编辑软件处理)")
    
    # 噪声不一致性检测
    total_tests += 1
    if results.get('noise_forgery') == True:
        suspicious_count += 1
        print("  - 检测到噪声不一致性 (可能经过合成或修改)")
    
    # CFA伪影检测 (如果有)
    if 'cfa_artifacts' in results:
        total_tests += 1
        if results.get('cfa_artifacts', 0) > 0:
            suspicious_count += 1
            print(f"  - 检测到{results['cfa_artifacts']}个CFA伪影 (可能经过复制粘贴操作)")
    
    # 复制-移动检测
    total_tests += 1
    if results.get('copy_move') == True:
        suspicious_count += 1
        print("  - 检测到复制-移动操作痕迹 (可能经过复制粘贴操作)")
        
    # 错误等级分析
    total_tests += 1
    if results.get('ela') == True:
        suspicious_count += 1
        print("  - 错误等级分析检测到异常 (可能经过编辑处理)")
        
    # 光照一致性分析
    total_tests += 1
    if results.get('illumination') == True:
        suspicious_count += 1
        print("  - 检测到光照不一致 (可能经过合成或修改)")
        
    # 边缘分析
    total_tests += 1
    if results.get('edge') == True:
        suspicious_count += 1
        print("  - 检测到异常边缘 (可能经过复制粘贴操作)")
        
    # 频域分析
    total_tests += 1
    if results.get('frequency') == True:
        suspicious_count += 1
        print("  - 频域分析检测到异常 (可能经过处理)")
        
    # 图像提取分析
    total_tests += 1
    if results.get('image_decode') == True:
        suspicious_count += 1
        print("  - 图像提取分析检测到隐藏信息 (可能包含隐藏数据)")
    
    # 综合判断
    print(f"\n综合分析:")
    print(f"  总检测项目: {total_tests}")
    print(f"  可疑项目: {suspicious_count}")
    
    if suspicious_count == 0:
        return "图像未发现明显篡改痕迹，可能是原始图像"
    elif suspicious_count <= total_tests * 0.3:
        return "图像发现少量可疑痕迹，篡改可能性较低"
    elif suspicious_count <= total_tests * 0.6:
        return "图像发现中等数量可疑痕迹，存在篡改可能性"
    else:
        return "图像发现大量可疑痕迹，很可能经过人为篡改"


def create_detection_folder():
    """创建用于存储检测结果的文件夹"""
    detection_folder = "./detection_results"
    if not path.exists(detection_folder):
        os.makedirs(detection_folder)
    return detection_folder


def process_single_image_fast(file_name):
    """快速处理单个图像文件（跳过耗时的CFA检测）"""
    input_path = './input/' + file_name
    if not path.exists(input_path):
        print("找不到图片: {}. 请将图片放在input子目录中.".format(file_name))
        return

    print(f"\n正在处理图片: {file_name}")
    PrintBoundary()
    
    # 创建检测结果文件夹
    detection_folder = create_detection_folder()
    
    # 存储检测结果
    results = {}

    # double jpeg compression detection Start
    PrintBoundary()
    print('\n运行双重JPEG压缩检测...')
    start_time = time.time()
    try:
        double_compressed = double_jpeg_compression.detect(input_path)
        results['double_compressed'] = double_compressed
        if(double_compressed):
            print('\n检测到双重压缩')
        else:
            print('\n单一压缩')
    except Exception as e:
        print(f'\n双重JPEG压缩检测出错: {e}')
        results['double_compressed'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # double jpeg compression detection End

    # Metadata Analysis detection Start
    PrintBoundary()
    print('\n运行元数据分析检测')
    start_time = time.time()
    try:
        img = Image.open(input_path)
        img_exif = img.getexif()

        if img_exif is None:
            print('抱歉，图片没有EXIF数据.')
            results['metadata'] = False
        else:
            results['metadata'] = True
            for key, val in img_exif.items():
                if key in ExifTags.TAGS:
                    print(f'{ExifTags.TAGS[key]} : {val}')
    except Exception as e:
        print(f'元数据分析出错: {e}')
        results['metadata'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # Metadata Analysis detection End

    # noise variance inconsistency detection Start
    PrintBoundary()
    print('\n运行噪声方差不一致性检测...')
    start_time = time.time()
    try:
        noise_forgery = noise_variance.detect(input_path)
        results['noise_forgery'] = noise_forgery
        if(noise_forgery):
            print('\n检测到噪声方差不一致')
        else:
            print('\n未检测到噪声方差不一致')
    except Exception as e:
        print(f'\n噪声方差不一致性检测出错: {e}')
        results['noise_forgery'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # noise variance inconsistency detection Start

    # Copy-Move detection Start
    eps = 60
    min_samples = 2

    PrintBoundary()
    print('\n运行复制-移动检测...')
    start_time = time.time()
    try:
        detect = Detect(input_path)
        key_points, descriptors = detect.siftDetector()
        forgery = detect.locateForgery(eps, min_samples)
        results['copy_move'] = forgery is not None
        if forgery is None:
            print("未发现伪造")
        else:
            print("检测到伪造区域")
            # 保存结果到检测文件夹
            name = re.findall(r'(.+?)(\.[^.]*$|$)', file_name)
            date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
            new_file_name = name[0][0]+'_'+str(eps)+'_'+str(min_samples)
            new_file_name = new_file_name+'_'+date+name[0][1]
            full_path = path.join(detection_folder, f"copy_move_{new_file_name}")
            PrintBoundary()

            success = cv2.imwrite(full_path, forgery)
            if success:
                print('伪造检测结果已保存为:', full_path)
            else:
                print('保存伪造检测结果失败')

    except Exception as e:
        print(f'复制-移动检测出错: {e}')
        results['copy_move'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    # Copy-Move detection End

    # 综合分析
    PrintBoundary()
    print('\n图像真实性分析:')
    conclusion = analyze_image_authenticity(results)
    print(f'  结论: {conclusion}')
    PrintBoundary()

    print(f"\n完成处理图片: {file_name}")
    PrintBoundary()
    PrintBoundary()


def process_single_image_all(file_name):
    """完整处理单个图像文件（包含所有检测方法）"""
    input_path = './input/' + file_name
    if not path.exists(input_path):
        print("找不到图片: {}. 请将图片放在input子目录中.".format(file_name))
        return

    print(f"\n正在处理图片: {file_name}")
    PrintBoundary()
    
    # 创建检测结果文件夹
    detection_folder = create_detection_folder()
    
    # 存储检测结果
    results = {}

    # 1. 双重JPEG压缩检测 Start
    PrintBoundary()
    print('\n运行双重JPEG压缩检测...')
    start_time = time.time()
    try:
        double_compressed = double_jpeg_compression.detect(input_path)
        results['double_compressed'] = double_compressed
        if(double_compressed):
            print('\n检测到双重压缩')
        else:
            print('\n单一压缩')
    except Exception as e:
        print(f'\n双重JPEG压缩检测出错: {e}')
        results['double_compressed'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 双重JPEG压缩检测 End

    # 2. 元数据分析 Start
    PrintBoundary()
    print('\n运行元数据分析检测')
    start_time = time.time()
    try:
        from PIL import Image, ExifTags
        img = Image.open(input_path)
        img_exif = img.getexif()

        if img_exif is None:
            print('抱歉，图片没有EXIF数据.')
            results['metadata'] = False
        else:
            results['metadata'] = True
            for key, val in img_exif.items():
                if key in ExifTags.TAGS:
                    print(f'{ExifTags.TAGS[key]} : {val}')
    except Exception as e:
        print(f'元数据分析出错: {e}')
        results['metadata'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 元数据分析 End

    # 3. CFA伪影检测 Start
    PrintBoundary()
    print('\n运行CFA伪影检测...')
    start_time = time.time()
    try:
        import copy_move_cfa
        identical_regions_cfa = copy_move_cfa.detect(input_path, opt, args)
        results['cfa_artifacts'] = identical_regions_cfa
        print('\n' + str(identical_regions_cfa), '个CFA伪影被检测到')
    except Exception as e:
        print(f'\nCFA伪影检测出错: {e}')
        results['cfa_artifacts'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # CFA伪影检测 End

    # 4. 噪声方差不一致性检测 Start
    PrintBoundary()
    print('\n运行噪声方差不一致性检测...')
    start_time = time.time()
    try:
        noise_forgery = noise_variance.detect(input_path)
        results['noise_forgery'] = noise_forgery
        if(noise_forgery):
            print('\n检测到噪声方差不一致')
        else:
            print('\n未检测到噪声方差不一致')
    except Exception as e:
        print(f'\n噪声方差不一致性检测出错: {e}')
        results['noise_forgery'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 噪声方差不一致性检测 End

    # 5. 复制-移动检测 Start
    eps = 60
    min_samples = 2

    PrintBoundary()
    print('\n运行复制-移动检测...')
    start_time = time.time()
    try:
        import cv2
        detect = Detect(input_path)
        key_points, descriptors = detect.siftDetector()
        forgery = detect.locateForgery(eps, min_samples)
        results['copy_move'] = forgery is not None
        if forgery is None:
            print("未发现伪造")
        else:
            print("检测到伪造区域")
            # 保存结果到检测文件夹
            name = re.findall(r'(.+?)(\.[^.]*$|$)', file_name)
            if name:
                date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
                new_file_name = name[0][0]+'_'+str(eps)+'_'+str(min_samples)
                new_file_name = new_file_name+'_'+date+name[0][1]
                full_path = path.join(detection_folder, f"copy_move_{new_file_name}")
                PrintBoundary()

                success = cv2.imwrite(full_path, forgery)
                if success:
                    print('伪造检测结果已保存为:', full_path)
                else:
                    print('保存伪造检测结果失败')

    except Exception as e:
        print(f'复制-移动检测出错: {e}')
        results['copy_move'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    # 复制-移动检测 End

    # 6. 错误等级分析检测 Start
    PrintBoundary()
    print('\n运行错误等级分析检测...')
    start_time = time.time()
    try:
        from PIL import Image, ImageChops
        import numpy as np
        
        # 打开图像
        original = Image.open(input_path)
        TEMP = 'temp.jpg'
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)

        diff = ImageChops.difference(original, temporary)
        d = diff.load()
        WIDTH, HEIGHT = diff.size
        for x in range(WIDTH):
            for y in range(HEIGHT):
                d[x, y] = tuple(k * 10 for k in d[x, y])  # 调整缩放因子以增强可见性

        # 保存结果到检测文件夹
        filename = file_name.split('.')[0]
        timestamp = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        ela_filename = f"ela_{filename}_{timestamp}.jpg"
        ela_full_path = path.join(detection_folder, ela_filename)
        
        diff.save(ela_full_path)
        print(f'错误等级分析检测结果已保存为: {ela_full_path}')
        results['ela'] = True
        
        # 清理临时文件
        if path.exists(TEMP):
            os.remove(TEMP)
            
    except Exception as e:
        print(f'错误等级分析检测出错: {e}')
        results['ela'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 错误等级分析检测 End

    # 7. 光照一致性分析检测 Start
    PrintBoundary()
    print('\n运行光照一致性分析检测...')
    start_time = time.time()
    try:
        import cv2
        import numpy as np
        
        # 加载图像
        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用Sobel算子计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # 简单的光照一致性检查
        # 如果图像中存在明显的光照不一致区域，角度分布会不均匀
        angle_hist, _ = np.histogram(angle, bins=36, range=(-np.pi, np.pi))
        
        # 计算直方图的方差
        angle_variance = np.var(angle_hist)
        
        # 根据方差判断是否存在光照不一致
        if angle_variance > 5000:  # 阈值需要根据实际情况调整
            print("检测到光照不一致")
            results['illumination'] = True
        else:
            print("光照一致性良好")
            results['illumination'] = False
            
        # 保存可视化结果
        filename = file_name.split('.')[0]
        timestamp = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        illumination_filename = f"illumination_{filename}_{timestamp}.jpg"
        illumination_full_path = path.join(detection_folder, illumination_filename)
        
        # 创建可视化图像（显示梯度幅值）
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(illumination_full_path, magnitude_normalized)
        print(f'光照一致性分析检测结果已保存为: {illumination_full_path}')
        
    except Exception as e:
        print(f'光照一致性分析检测出错: {e}')
        results['illumination'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 光照一致性分析检测 End

    # 8. 边缘分析检测 Start
    PrintBoundary()
    print('\n运行边缘分析检测...')
    start_time = time.time()
    try:
        import cv2
        import numpy as np
        
        # 加载图像
        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        # 计算边缘密度
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 根据边缘特征判断是否存在篡改痕迹
        if lines is not None and len(lines) > 100:  # 阈值需要根据实际情况调整
            print("检测到异常边缘，可能存在篡改")
            results['edge'] = True
        else:
            print("边缘特征正常")
            results['edge'] = False
            
        # 保存边缘检测结果
        filename = file_name.split('.')[0]
        timestamp = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        edge_filename = f"edge_{filename}_{timestamp}.jpg"
        edge_full_path = path.join(detection_folder, edge_filename)
        cv2.imwrite(edge_full_path, edges)
        print(f'边缘分析检测结果已保存为: {edge_full_path}')
        
    except Exception as e:
        print(f'边缘分析检测出错: {e}')
        results['edge'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 边缘分析检测 End

    # 9. 频域分析检测 Start
    PrintBoundary()
    print('\n运行频域分析检测...')
    start_time = time.time()
    try:
        import cv2
        import numpy as np
        
        # 加载图像
        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 执行FFT变换
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        
        # 计算频谱的统计特征
        mean_spectrum = np.mean(magnitude_spectrum)
        std_spectrum = np.std(magnitude_spectrum)
        
        # 根据频谱特征判断是否存在异常
        # 这里使用一个简单的阈值判断，实际应用中需要更复杂的分析
        if std_spectrum > 50:  # 阈值需要根据实际情况调整
            print("频域分析检测到异常")
            results['frequency'] = True
        else:
            print("频域特征正常")
            results['frequency'] = False
            
        # 保存频谱图像
        filename = file_name.split('.')[0]
        timestamp = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        frequency_filename = f"frequency_{filename}_{timestamp}.jpg"
        frequency_full_path = path.join(detection_folder, frequency_filename)
        
        # 归一化频谱图像
        magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(frequency_full_path, magnitude_spectrum_normalized)
        print(f'频域分析检测结果已保存为: {frequency_full_path}')
        
    except Exception as e:
        print(f'频域分析检测出错: {e}')
        results['frequency'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 频域分析检测 End

    # 10. 图像提取分析 Start
    PrintBoundary()
    print('\n运行图像提取分析...')
    start_time = time.time()
    try:
        import cv2
        import numpy as np
        import random
        
        # 加密图像
        img = cv2.imread(input_path) 
        width = img.shape[0]
        height = img.shape[1]
        
        # img1 and img2 are two blank images
        img1 = np.zeros((width, height, 3), np.uint8)
        img2 = np.zeros((width, height, 3), np.uint8)
        
        for i in range(width):
            for j in range(height):
                for l in range(3):
                    v1 = format(img[i][j][l], '08b')
                    v2 = v1[:4] + chr(random.randint(0, 1)+48) * 4
                    v3 = v1[4:] + chr(random.randint(0, 1)+48) * 4
                    
                    # Appending data to img1 and img2
                    img1[i][j][l]= int(v2, 2)
                    img2[i][j][l]= int(v3, 2)
        
        # 保存结果到检测文件夹
        filename = file_name.split('.')[0]
        timestamp = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        decode_filename1 = f"decode1_{filename}_{timestamp}.jpg"
        decode_filename2 = f"decode2_{filename}_{timestamp}.jpg"
        decode_full_path1 = path.join(detection_folder, decode_filename1)
        decode_full_path2 = path.join(detection_folder, decode_filename2)
        
        cv2.imwrite(decode_full_path1, img1)
        cv2.imwrite(decode_full_path2, img2)
        print(f'图像提取分析结果已保存为: {decode_full_path1} 和 {decode_full_path2}')
        results['image_decode'] = True
        
    except Exception as e:
        print(f'图像提取分析出错: {e}')
        results['image_decode'] = None
    end_time = time.time()
    print(f'耗时: {end_time - start_time:.2f} 秒')
    PrintBoundary()
    # 图像提取分析 End

    # 综合分析
    PrintBoundary()
    print('\n图像真实性分析:')
    conclusion = analyze_image_authenticity(results)
    print(f'  结论: {conclusion}')
    PrintBoundary()

    print(f"\n完成处理图片: {file_name}")
    PrintBoundary()
    PrintBoundary()


def process_all_images(mode="fast"):
    """处理input文件夹中的所有图像文件"""
    input_dir = './input'
    if not path.exists(input_dir):
        print("找不到input目录")
        return

    # 支持的图像格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("在input目录中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像文件
    for i, file_name in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理文件: {file_name}")
        if mode == "fast":
            process_single_image_fast(file_name)
        else:
            process_single_image_all(file_name)
    
    print(f"\n所有 {len(image_files)} 个图像文件处理完成!")


def main():
    try:
        # 检查命令行参数
        if len(sys.argv) > 1:
            mode = sys.argv[1]  # "fast" 或 "all"
            if mode not in ["fast", "all"]:
                print("使用方法: python batch_process.py [fast|all]")
                print("  fast - 快速模式（跳过耗时的CFA检测）")
                print("  all  - 完整模式（包含所有检测方法）")
                return
        else:
            mode = "fast"  # 默认使用快速模式
            
        print(f"使用 {mode} 模式批量处理input文件夹中的所有图像...")
        process_all_images(mode)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()