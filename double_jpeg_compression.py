import numpy as np
# import pandas as pd
import cv2
# import argparse
# import csv
# import sys
import os
from datetime import datetime

from scipy import fftpack as fftp
from matplotlib import pyplot as plt


def create_detection_folder():
    """创建用于存储检测结果的文件夹"""
    detection_folder = "./detection_results"
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)
    return detection_folder


def detect(image_path):
    try:
        firstq = 30
        secondq = 40
        thres = 0.5

        dct_rows = 0
        dct_cols = 0

        image = cv2.imread(image_path)
        if image is None:
            print("无法加载图像")
            return False
            
        shape = image.shape

        if shape[0] % 8 != 0:
            dct_rows = shape[0]+8-shape[0] % 8
        else:
            dct_rows = shape[0]

        if shape[1] % 8 != 0:
            dct_cols = shape[1]+8-shape[1] % 8
        else:
            dct_cols = shape[1]

        dct_image = np.zeros((dct_rows, dct_cols, 3), np.uint8)
        dct_image[0:shape[0], 0:shape[1]] = image

        y = cv2.cvtColor(dct_image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

        w = y.shape[1]
        h = y.shape[0]
        n = w*h/64

        Y = y.reshape(h//8, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8)

        qDCT = []

        for i in range(0, Y.shape[0]):
            qDCT.append(cv2.dct(np.float32(Y[i])))

        qDCT = np.asarray(qDCT, dtype=np.float32)
        qDCT = np.rint(qDCT - np.mean(qDCT, axis=0)).astype(np.int32)
        f, a1 = plt.subplots(8, 8)
        a1 = a1.ravel()

        k = 0
        # flag = True
        for idx, ax in enumerate(a1):
            k += 1
            data = qDCT[:, int(idx/8), int(idx % 8)]
            val, key = np.histogram(data, bins=np.arange(data.min(), data.max()+1))
            # val, key = np.histogram(data, bins=np.arange(data.min(), data.max()+1), normed=True)
            z = np.absolute(fftp.fft(val))
            z = np.reshape(z, (len(z), 1))
            rotz = np.roll(z, int(len(z)/2))

            slope = rotz[1:] - rotz[:-1]
            indices = [i+1 for i in range(len(slope)-1)
                    if slope[i] > 0 and slope[i+1] < 0]

            peak_count = 0

            for j in indices:
                if rotz[j][0] > thres:
                    peak_count += 1

            if(k==3):
                if peak_count>=20: 
                    # 保存图像到检测结果文件夹
                    detection_folder = create_detection_folder()
                    filename = os.path.basename(image_path).split('.')[0]
                    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                    save_path = os.path.join(detection_folder, f"double_jpeg_{filename}_{timestamp}.png")
                    plt.savefig(save_path)
                    plt.close(f)  # Close the figure to free memory
                    print(f'双重JPEG压缩检测结果已保存为: {save_path}')
                    return True
                else: 
                    # 保存图像到检测结果文件夹
                    detection_folder = create_detection_folder()
                    filename = os.path.basename(image_path).split('.')[0]
                    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                    save_path = os.path.join(detection_folder, f"double_jpeg_{filename}_{timestamp}.png")
                    plt.savefig(save_path)
                    plt.close(f)  # Close the figure to free memory
                    print(f'双重JPEG压缩检测结果已保存为: {save_path}')
                    return False
                # flag = False
    except Exception as e:
        print(f"Error in double JPEG compression detection: {e}")
        return False