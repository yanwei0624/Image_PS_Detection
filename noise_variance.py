import sys
import math
import numpy as np
import os
from datetime import datetime

from PIL import Image
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def create_detection_folder():
    """创建用于存储检测结果的文件夹"""
    detection_folder = "./detection_results"
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)
    return detection_folder

def estimate_noise(I):
    H, W = I.shape

    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma

def detect(input, blockSize=32):
    try:
        im = Image.open(input)
        im = im.convert('1')
    except Exception as e:
        print(f"Error opening image: {e}")
        return False

    blocks = []

    imgwidth, imgheight = im.size

    # break up image into NxN blocks, N = blockSize
    for i in range(0,imgheight,blockSize):
        for j in range(0,imgwidth,blockSize):
            box = (j, i, j+blockSize, i+blockSize)
            b = im.crop(box)
            a = np.asarray(b).astype(int)
            blocks.append(a)

    if len(blocks) == 0:
        return False

    variances = []
    for block in blocks:
        variances.append([estimate_noise(block)])

    try:
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(variances)
        center1, center2 = kmeans.cluster_centers_

        # 创建可视化图表
        plt.figure(figsize=(10, 6))
        variances_flat = [v[0] for v in variances]  # 展平数据
        plt.hist(variances_flat, bins=30, alpha=0.7, color='blue')
        plt.axvline(center1, color='red', linestyle='--', label=f'Cluster 1 Center: {center1[0]:.2f}')
        plt.axvline(center2, color='green', linestyle='--', label=f'Cluster 2 Center: {center2[0]:.2f}')
        plt.xlabel('Noise Variance')
        plt.ylabel('Frequency')
        plt.title('Noise Variance Distribution')
        plt.legend()
        
        # 保存图像到检测结果文件夹
        detection_folder = create_detection_folder()
        filename = os.path.basename(input).split('.')[0]
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        save_path = os.path.join(detection_folder, f"noise_variance_{filename}_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()
        print(f'噪声方差检测结果已保存为: {save_path}')

        if abs(center1 - center2) > .4: 
            return True
        else: 
            return False
    except Exception as e:
        print(f"Error in KMeans clustering: {e}")
        return False