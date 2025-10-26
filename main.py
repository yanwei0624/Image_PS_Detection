import sys
import cv2
from ForgeryDetection import Detect
import re
from datetime import datetime
import os.path as path
# from exif import Image

from PIL import Image, ExifTags
import os

import double_jpeg_compression
import copy_move_cfa
import noise_variance

from optparse import OptionParser

# copy-move parameters
cmd = OptionParser("使用方法: %prog image_file [options]")
cmd.add_option('', '--imauto',
               help='自动搜索相同区域. (默认: %default)', default=1)
cmd.add_option('', '--imblev',
               help='图像细节模糊级别. (默认: %default)', default=8)
cmd.add_option('', '--impalred',
               help='图像调色板减少因子. (默认: %default)', default=15)
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
# if not args:
#     cmd.print_help()
#     sys.exit()


def PrintBoundary():
    for i in range(50):
        print('*', end='')
    print()


def process_single_image(file_name):
    """处理单个图像文件"""
    input_path = './/input//' + file_name
    if not path.exists(input_path):
        print("找不到图片: {}. 请将图片放在input子目录中.".format(file_name))
        return

    print(f"\n正在处理图片: {file_name}")
    PrintBoundary()

    # double jpeg compression detection Start
    PrintBoundary()
    print('\n运行双重JPEG压缩检测...')
    try:
        double_compressed = double_jpeg_compression.detect(input_path)
        if(double_compressed):
            print('\n检测到双重压缩')
        else:
            print('\n单一压缩')
    except Exception as e:
        print(f'\n双重JPEG压缩检测出错: {e}')
    PrintBoundary()
    # double jpeg compression detection End

    # Metadata Analysis detection Start
    PrintBoundary()
    print('\n运行元数据分析检测')
    try:
        img = Image.open(input_path)
        img_exif = img.getexif()

        if img_exif is None:
            print('抱歉，图片没有EXIF数据.')
        else:
            for key, val in img_exif.items():
                if key in ExifTags.TAGS:
                    print(f'{ExifTags.TAGS[key]} : {val}')
    except Exception as e:
        print(f'元数据分析出错: {e}')
    PrintBoundary()
    # Metadata Analysis detection End

    # CFA artifact detection Start
    PrintBoundary()
    print('\n运行CFA伪影检测...\n')
    try:
        identical_regions_cfa = copy_move_cfa.detect(input_path, opt, args)
        print('\n' + str(identical_regions_cfa), '个CFA伪影被检测到')
    except Exception as e:
        print(f'\nCFA伪影检测出错: {e}')
    PrintBoundary()
    # CFA artifact detection End

    # noise variance inconsistency detection Start
    PrintBoundary()
    print('\n运行噪声方差不一致性检测...')
    try:
        noise_forgery = noise_variance.detect(input_path)
        if(noise_forgery):
            print('\n检测到噪声方差不一致')
        else:
            print('\n未检测到噪声方差不一致')
    except Exception as e:
        print(f'\n噪声方差不一致性检测出错: {e}')
    PrintBoundary()
    # noise variance inconsistency detection Start

    # Copy-Move detection Start
    eps = 60
    min_samples = 2

    PrintBoundary()
    print('使用 \'q\' 退出程序\n使用 \'s/S\' 保存伪造检测结果.')
    PrintBoundary()
    
    PrintBoundary()
    print('使用参数值检测复制-移动伪造\neps:{}\nmin_samples:{}'.format(
        eps, min_samples))
    PrintBoundary()

    try:
        detect = Detect(input_path)
        key_points, descriptors = detect.siftDetector()
        forgery = detect.locateForgery(eps, min_samples)
        if forgery is None:
            print("未发现伪造")
        else:
            print("检测到伪造区域")
            # 保存结果
            name = re.findall(r'(.+?)(\.[^.]*$|$)', file_name)
            date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
            new_file_name = name[0][0]+'_'+str(eps)+'_'+str(min_samples)
            new_file_name = new_file_name+'_'+date+name[0][1]
            PrintBoundary()

            success = cv2.imwrite(new_file_name, forgery)
            if success:
                print('伪造检测结果已保存为:', new_file_name)
            else:
                print('保存伪造检测结果失败')

    except Exception as e:
        print(f'复制-移动检测出错: {e}')
    # Copy-Move detection End

    print(f"\n完成处理图片: {file_name}")
    PrintBoundary()
    PrintBoundary()


def process_all_images():
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
        process_single_image(file_name)
    
    print(f"\n所有 {len(image_files)} 个图像文件处理完成!")


def main():
    try:
        # 检查是否有命令行参数
        if len(sys.argv) > 1:
            # 如果有参数，按照原来的方式处理单个文件
            file_name = sys.argv[1]
            process_single_image(file_name)
        else:
            # 如果没有参数，批量处理所有图像
            print("未指定图像文件，将批量处理input文件夹中的所有图像...")
            process_all_images()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()