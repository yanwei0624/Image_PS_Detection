from importlib.resources import path
from tkinter import *
from tkinter import filedialog, ttk, messagebox
from PIL import ImageTk, Image, ExifTags, ImageChops
from optparse import OptionParser
from datetime import datetime
from matplotlib import image
from prettytable import PrettyTable
import numpy as np
import random
import sys
import cv2
import re
import os

from pyparsing import Opt

from ForgeryDetection import Detect
import double_jpeg_compression
import noise_variance
import copy_move_cfa


# Global variables
IMG_WIDTH = 200
IMG_HEIGHT = 200
uploaded_image = None
original_image = None

# copy-move parameters
cmd = OptionParser("usage: %prog image_file [options]")
cmd.add_option('', '--imauto',
               help='Automatically search identical regions. (default: %default)', default=1)
cmd.add_option('', '--imblev',
               help='Blur level for degrading image details. (default: %default)', default=8)
cmd.add_option('', '--impalred',
               help='Image palette reduction factor. (default: %default)', default=15)
cmd.add_option(
    '', '--rgsim', help='Region similarity threshold. (default: %default)', default=5)
cmd.add_option(
    '', '--rgsize', help='Region size threshold. (default: %default)', default=1.5)
cmd.add_option(
    '', '--blsim', help='Block similarity threshold. (default: %default)', default=200)
cmd.add_option('', '--blcoldev',
               help='Block color deviation threshold. (default: %default)', default=0.2)
cmd.add_option(
    '', '--blint', help='Block intersection threshold. (default: %default)', default=0.2)
opt, args = cmd.parse_args()
# if not args:
#     cmd.print_help()
#     sys.exit()


def getImage(path, width, height):
    """
    Function to return an image as a PhotoImage object
    :param path: A string representing the path of the image file
    :param width: The width of the image to resize to
    :param height: The height of the image to resize to
    :return: The image represented as a PhotoImage object
    """
    img = Image.open(path)
    img = img.resize((width, height), Image.LANCZOS)

    return ImageTk.PhotoImage(img)


def browseFile():
    """
    Function to open a browser for users to select an image
    :return: None
    """
    # Only accept jpg and png files
    filename = filedialog.askopenfilename(title="选择图片", filetypes=[("image", ".jpeg"),("image", ".png"),("image", ".jpg")])

    # No file selected (User closes the browsing window)
    if filename == "":
        return

    global uploaded_image, original_image

    uploaded_image = filename
    original_image = Image.open(filename)

    progressBar['value'] = 0   # Reset the progress bar
    fileLabel.configure(text=filename)     # Set the path name in the fileLabel

    # Display the input image in imagePanel
    img = getImage(filename, IMG_WIDTH, IMG_HEIGHT)
    imagePanel.configure(image=img)
    imagePanel.image = img

    # Display blank image in resultPanel
    blank_img = getImage("images/output.png", IMG_WIDTH, IMG_HEIGHT)
    resultPanel.configure(image=blank_img)
    resultPanel.image = blank_img

    # Reset the resultLabel
    resultLabel.configure(text="准备扫描", foreground="green")


def copy_move_forgery():
    # Retrieve the path of the image file
    path = uploaded_image
    eps = 60
    min_samples = 2

    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        detect = Detect(path)
        key_points, descriptors = detect.siftDetector()
        forgery = detect.locateForgery(eps, min_samples)

        # Set the progress bar to 100%
        progressBar['value'] = 100

        if forgery is None:
            # Retrieve the thumbs up image and display in resultPanel
            img = getImage("images/no_copy_move.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="原图", foreground="green")
        else:
            # Retrieve the output image and display in resultPanel
            img = getImage("images/copy_move.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="图像被篡改", foreground="red")
            # cv2.imshow('Original image', detect.image)
            cv2.imshow('伪造检测结果', forgery)
            wait_time = 1000
            while(cv2.getWindowProperty('Forgery', 0) >= 0) or (cv2.getWindowProperty('Original image', 0) >= 0):
                keyCode = cv2.waitKey(wait_time)
                if (keyCode) == ord('q') or keyCode == ord('Q'):
                    cv2.destroyAllWindows()
                    break
                elif keyCode == ord('s') or keyCode == ord('S'):
                    # 创建检测结果文件夹
                    detection_folder = "detection_results"
                    if not os.path.exists(detection_folder):
                        os.makedirs(detection_folder)
                    
                    name = re.findall(r'(.+?)(\.[^.]*$|$)', path)
                    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
                    new_file_name = name[0][0]+'_'+str(eps)+'_'+str(min_samples)
                    new_file_name = new_file_name+'_'+date+name[0][1]
                    
                    # 保存到检测结果文件夹
                    full_path = os.path.join(detection_folder, f"copy_move_{new_file_name}")
                    value = cv2.imwrite(full_path, forgery)
                    print('图片保存为....', full_path)
                    messagebox.showinfo('保存成功', f'图片已保存为: {full_path}')
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return


def metadata_analysis():
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        img = Image.open(path)
        img_exif = img.getexif()

        # Set the progress bar to 100%
        progressBar['value'] = 100

        if img_exif is None:
            # print('Sorry, image has no exif data.')
            # Retrieve the output image and display in resultPanel
            img = getImage("images/no_metadata.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="未找到数据", foreground="red")
        else:
            # Retrieve the thumbs up image and display in resultPanel
            img = getImage("images/metadata.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="元数据详情", foreground="green")

            # print('image has exif data.')
            with open('Metadata_analysis.txt', 'w') as f:
                for key, val in img_exif.items():
                    if key in ExifTags.TAGS:
                        # print(f'{ExifTags.TAGS[key]} : {val}')
                            f.write(f'{ExifTags.TAGS[key]} : {val}\n')
            os.startfile('Metadata_analysis.txt')
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return


def noise_variance_inconsistency():
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        noise_forgery = noise_variance.detect(path)

        # Set the progress bar to 100%
        progressBar['value'] = 100

        if(noise_forgery):
            # print('\nNoise variance inconsistency detected')
            # Retrieve the output image and display in resultPanel
            img = getImage("images/varience.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="噪声方差不一致", foreground="red")
        
        else:
            # Retrieve the thumbs up image and display in resultPanel
            img = getImage("images/no_varience.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="无噪声方差不一致", foreground="green")
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return

def cfa_artifact():
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        identical_regions_cfa = copy_move_cfa.detect(path, opt, args)
        # identical_regions_cfa = copy_move_cfa.detect(path, opt, args)


        # Set the progress bar to 100%
        progressBar['value'] = 100

        # print('\n' + str(identical_regions_cfa), 'CFA artifacts detected')

        if(identical_regions_cfa):
            # Retrieve the output image and display in resultPanel
            img = getImage("images/cfa.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text=f"{str(identical_regions_cfa)}个CFA伪影被检测到", foreground="red")

        else:
            # print('\nSingle compressed')
            # Retrieve the thumbs up image and display in resultPanel
            img = getImage("images/no_cfa.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="未检测到CFA伪影", foreground="green")
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return


def ela_analysis():
    # Retrieve the path of the image file
    path = uploaded_image
    TEMP = 'temp.jpg'
    SCALE = 10

    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        original = Image.open(path)
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)

        diff = ImageChops.difference(original, temporary)
        d = diff.load()
        WIDTH, HEIGHT = diff.size
        for x in range(WIDTH):
            for y in range(HEIGHT):
                d[x, y] = tuple(k * SCALE for k in d[x, y])

        # Set the progress bar to 100%
        progressBar['value'] = 100
        diff.show()
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return


def jpeg_Compression():

    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        double_compressed = double_jpeg_compression.detect(path)

        # Set the progress bar to 100%
        progressBar['value'] = 100

        if(double_compressed):
            # print('\nDouble compression detected')
            # Retrieve the output image and display in resultPanel
            img = getImage("images/double_compression.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="双重JPEG压缩", foreground="red")

        else:
            # print('\nSingle compressed')
            # Retrieve the thumbs up image and display in resultPanel
            img = getImage("images/single_compression.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img)
            resultPanel.image = img

            # Display results in resultLabel
            resultLabel.configure(text="单一JPEG压缩", foreground="green")
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return

def image_decode():
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return
    
    try:
        # Encrypted image
        img = cv2.imread(path) 
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
        
        # Set the progress bar to 100%
        progressBar['value'] = 100

        # These are two images produced from
        # the encrypted image
        # cv2.imwrite('pic2_re.png', img1)
        cv2.imwrite('output.png', img2)
        # Image.show(img2)
        # creating a object
        im = Image.open('output.png')
        im.show()
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return

def string_analysis():
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return
    
    try:
        x=PrettyTable()
        x.field_names = ["字节", "8位", "字符串"]
        # x.border = False
        with open(path, "rb") as f:
                n = 0
                b = f.read(16)

                while b:
                    s1 = " ".join([f"{i:02x}" for i in b])  # hex string
                    # insert extra space between groups of 8 hex values
                    s1 = s1[0:23] + " " + s1[23:]

                    # ascii string; chained comparison
                    s2 = "".join([chr(i) if 32 <= i <= 127 else "." for i in b])

                    # print(f"{n * 16:08x}  {s1:<48}  |{s2}|")
                    x.add_row([f"{n * 16:08x}",f"{s1:<48}",f"{s2}"])

                    n += 1
                    b = f.read(16)
                
                # Set the progress bar to 100%
                progressBar['value'] = 100

                with open('hex_viewer.txt', 'w') as w:
                    w.write(str(x))
                    # w.write(f"{os.path.getsize(path):08x}")
                os.startfile('hex_viewer.txt')
                # print(f"{os.path.getsize(filename):08x}")
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return

def illumination_analysis():
    """
    光照一致性分析检测
    """
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        # Load image
        img = cv2.imread(path)
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
        
        # Set the progress bar to 100%
        progressBar['value'] = 100
        
        # 根据方差判断是否存在光照不一致
        if angle_variance > 5000:  # 阈值需要根据实际情况调整
            # Retrieve the output image and display in resultPanel
            img_result = getImage("images/no_metadata.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img_result)
            resultPanel.image = img_result

            # Display results in resultLabel
            resultLabel.configure(text="检测到光照不一致", foreground="red")
        else:
            # Retrieve the thumbs up image and display in resultPanel
            img_result = getImage("images/metadata.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img_result)
            resultPanel.image = img_result

            # Display results in resultLabel
            resultLabel.configure(text="光照一致性良好", foreground="green")
            
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return

def edge_analysis():
    """
    边缘分析检测
    """
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        # Load image
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        # 计算边缘密度
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Set the progress bar to 100%
        progressBar['value'] = 100
        
        # 根据边缘特征判断是否存在篡改痕迹
        if lines is not None and len(lines) > 100:  # 阈值需要根据实际情况调整
            # Retrieve the output image and display in resultPanel
            img_result = getImage("images/varience.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img_result)
            resultPanel.image = img_result

            # Display results in resultLabel
            resultLabel.configure(text=f"检测到异常边缘，可能存在篡改", foreground="red")
        else:
            # Retrieve the thumbs up image and display in resultPanel
            img_result = getImage("images/no_varience.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img_result)
            resultPanel.image = img_result

            # Display results in resultLabel
            resultLabel.configure(text="边缘特征正常", foreground="green")
            
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return

def frequency_analysis():
    """
    频域分析检测
    """
    # Retrieve the path of the image file
    path = uploaded_image
    # User has not selected an input image
    if path is None:
        # Show error message
        messagebox.showerror('错误', "请选择图片")
        return

    try:
        # Load image
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 执行FFT变换
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        
        # 计算频谱的统计特征
        mean_spectrum = np.mean(magnitude_spectrum)
        std_spectrum = np.std(magnitude_spectrum)
        
        # Set the progress bar to 100%
        progressBar['value'] = 100
        
        # 根据频谱特征判断是否存在异常
        # 这里使用一个简单的阈值判断，实际应用中需要更复杂的分析
        if std_spectrum > 50:  # 阈值需要根据实际情况调整
            # Retrieve the output image and display in resultPanel
            img_result = getImage("images/cfa.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img_result)
            resultPanel.image = img_result

            # Display results in resultLabel
            resultLabel.configure(text=f"频域分析检测到异常", foreground="red")
        else:
            # Retrieve the thumbs up image and display in resultPanel
            img_result = getImage("images/no_cfa.png", IMG_WIDTH, IMG_HEIGHT)
            resultPanel.configure(image=img_result)
            resultPanel.image = img_result

            # Display results in resultLabel
            resultLabel.configure(text="频域特征正常", foreground="green")
            
    except Exception as e:
        messagebox.showerror('错误', f"处理图像时出错: {str(e)}")
        progressBar['value'] = 0
        return

def compare_images():
    """
    Function to compare original image with processed image
    :return: None
    """
    global original_image, uploaded_image
    
    if original_image is None or uploaded_image is None:
        messagebox.showerror('错误', "请选择图片")
        return
    
    try:
        # Create a new window for image comparison
        compare_window = Toplevel(root)
        compare_window.title("图像对比")
        
        # Get processed image (displayed in resultPanel)
        processed_img = Image.open("output.png") if os.path.exists("output.png") else original_image
        
        # Resize images for display
        original_resized = original_image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        processed_resized = processed_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        
        # Convert to PhotoImage
        original_tk = ImageTk.PhotoImage(original_resized)
        processed_tk = ImageTk.PhotoImage(processed_resized)
        
        # Display original image
        original_label = Label(compare_window, text="原始图像", font=("Courier", 15))
        original_label.grid(row=0, column=0, padx=10)
        original_panel = Label(compare_window, image=original_tk)
        original_panel.image = original_tk
        original_panel.grid(row=1, column=0, padx=10, pady=10)
        
        # Display processed image
        processed_label = Label(compare_window, text="处理后图像", font=("Courier", 15))
        processed_label.grid(row=0, column=1, padx=10)
        processed_panel = Label(compare_window, image=processed_tk)
        processed_panel.image = processed_tk
        processed_panel.grid(row=1, column=1, padx=10, pady=10)
        
        # Add difference image
        if processed_img != original_image:
            diff = ImageChops.difference(original_image, processed_img)
            diff_resized = diff.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
            diff_tk = ImageTk.PhotoImage(diff_resized)
            
            diff_label = Label(compare_window, text="差异图像", font=("Courier", 15))
            diff_label.grid(row=2, column=0, columnspan=2, pady=(20, 5))
            diff_panel = Label(compare_window, image=diff_tk)
            diff_panel.image = diff_tk
            diff_panel.grid(row=3, column=0, columnspan=2, pady=10)
        
    except Exception as e:
        messagebox.showerror('错误', f"图像对比出错: {str(e)}")


def run_all_detections():
    """一键运行所有检测方法"""
    global uploaded_image
    
    # 检查是否已选择图片
    if uploaded_image is None:
        messagebox.showerror('错误', "请选择图片")
        return
    
    try:
        # 显示处理中提示
        messagebox.showinfo('开始处理', "即将开始一键全部检测，这可能需要一些时间，请耐心等待...")
        
        # 创建新窗口显示进度
        progress_window = Toplevel(root)
        progress_window.title("一键检测进度")
        progress_window.geometry("400x300")
        
        # 添加滚动文本框显示进度
        text_frame = Frame(progress_window)
        text_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = Scrollbar(text_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        progress_text = Text(text_frame, wrap=WORD, yscrollcommand=scrollbar.set)
        progress_text.pack(fill=BOTH, expand=True)
        scrollbar.config(command=progress_text.yview)
        
        # 添加关闭按钮
        close_button = ttk.Button(progress_window, text="关闭", 
                                 command=progress_window.destroy)
        close_button.pack(pady=5)
        
        # 更新进度窗口
        progress_window.update()
        
        # 重定向输出到文本框
        import io
        import sys
        from contextlib import redirect_stdout
        
        # 创建StringIO对象捕获输出
        captured_output = io.StringIO()
        
        # 定义在新进程中运行的函数
        def run_detection_in_thread():
            try:
                # 运行批处理脚本的完整模式
                import subprocess
                import sys
                import os
                
                # 获取当前目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                batch_script = os.path.join(current_dir, "batch_process.py")
                
                # 运行批处理脚本的完整模式，实时显示输出
                process = subprocess.Popen([sys.executable, batch_script, "all"], 
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                         cwd=current_dir, text=True, bufsize=1, 
                                         universal_newlines=True)
                
                # 实时读取输出
                for line in process.stdout:
                    progress_text.insert(END, line)
                    progress_text.see(END)
                    progress_text.update_idletasks()
                
                process.wait()
                
                progress_text.insert(END, "\n一键检测完成!")
                progress_text.see(END)
                progress_text.config(state=DISABLED)
                
            except Exception as e:
                progress_text.insert(END, f"\n运行出错: {str(e)}")
                progress_text.see(END)
                progress_text.config(state=DISABLED)
        
        # 在新线程中运行检测
        import threading
        detection_thread = threading.Thread(target=run_detection_in_thread)
        detection_thread.daemon = True
        detection_thread.start()
        
    except Exception as e:
        messagebox.showerror('错误', f"启动一键检测时出错: {str(e)}")

# Initialize the app window
root = Tk()
root.title("图像篡改检测器")
root.iconbitmap('images/favicon.ico')

# Ensure the program closes when window is closed
root.protocol("WM_DELETE_WINDOW", root.quit)

# Maximize the size of the window
root.state("zoomed")

# Configure grid weights for responsive design
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)
root.rowconfigure(0, weight=0)  
root.rowconfigure(1, weight=0)  
root.rowconfigure(2, weight=0)  
root.rowconfigure(3, weight=0)  
root.rowconfigure(4, weight=0)  
root.rowconfigure(5, weight=0)  # all_detection button
root.rowconfigure(6, weight=1)  # button_frame
root.rowconfigure(7, weight=0)  # tools_frame
root.rowconfigure(8, weight=0)  # quit button

# Add the GUI into the Tkinter window
# GUI(parent=root)

# Label for the results of scan
resultLabel = Label(text="图像篡改检测系统", font=("Courier", 24))
resultLabel.grid(row=0, column=0, columnspan=3, pady=5)

# Get the blank image
input_img = getImage("images/input.png", 200, 200)
middle_img = getImage("images/middle.png", 200, 200)
output_img = getImage("images/output.png", 200, 200)

# Displays the input image
imagePanel = Label(image=input_img)
imagePanel.image = input_img
imagePanel.grid(row=1, column=0, padx=5)

# Label to display the middle image
middle = Label(image=middle_img)
middle.image = middle_img
middle.grid(row=1, column=1, padx=5)

# Label to display the output image
resultPanel = Label(image=output_img)
resultPanel.image = output_img
resultPanel.grid(row=1, column=2, padx=5)

# Label to display the path of the input image
fileLabel = Label(text="未选择文件", fg="grey", font=("Times", 15))
fileLabel.grid(row=2, column=0, columnspan=3, pady=5)

# Progress bar
progressBar = ttk.Progressbar(length=500)
progressBar.grid(row=3, column=0, columnspan=3, pady=10)

# Configure the style of the buttons
s = ttk.Style()
s.configure('my.TButton', font=('Times', 12))

# Button to upload images
uploadButton = ttk.Button(
    text="上传图片", style="my.TButton", command=browseFile)
uploadButton.grid(row=4, column=0, columnspan=3, pady=10)



# Button to run all detection methods
all_detection = ttk.Button(text="一键全部检测", style="my.TButton", command=run_all_detections)
all_detection.grid(row=5, column=0, columnspan=3, pady=5)

# Frame for detection buttons
button_frame = Frame(root, relief=RIDGE, bd=2)
button_frame.grid(row=6, column=0, columnspan=3, pady=10, padx=20, sticky="nsew")
button_frame.rowconfigure(0, weight=1)
button_frame.rowconfigure(1, weight=1)
button_frame.rowconfigure(2, weight=1)
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)
button_frame.columnconfigure(2, weight=1)

# Button to run the Compression detection algorithm
compression = ttk.Button(button_frame, text="JPEG压缩检测",
                         style="my.TButton", command=jpeg_Compression)
compression.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# Button to run the Metadata-Analysis detection algorithm
metadata = ttk.Button(button_frame, text="元数据分析",
                      style="my.TButton", command=metadata_analysis)
metadata.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

# Button to run the CFA-artifact detection algorithm
artifact = ttk.Button(button_frame, text="CFA伪影检测", style="my.TButton", command=cfa_artifact)
artifact.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

# Button to run the noise variance inconsistency detection algorithm
noise = ttk.Button(button_frame, text="噪声不一致性检测",
                   style="my.TButton", command=noise_variance_inconsistency)
noise.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

# Button to run the Copy-Move  detection algorithm
copy_move = ttk.Button(button_frame, text="复制-移动检测", style="my.TButton", command=copy_move_forgery)
copy_move.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

# Button to run the Error-Level Analysis algorithm
ela = ttk.Button(button_frame, text="错误等级分析", style="my.TButton", command=ela_analysis)
ela.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

# Button to run the Illumination Analysis algorithm
illumination = ttk.Button(button_frame, text="光照一致性分析", style="my.TButton", command=illumination_analysis)
illumination.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

# Button to run the Edge Analysis algorithm
edge_btn = ttk.Button(button_frame, text="边缘分析检测", style="my.TButton", command=edge_analysis)
edge_btn.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

# Button to run the Frequency Analysis algorithm
frequency_btn = ttk.Button(button_frame, text="频域分析检测", style="my.TButton", command=frequency_analysis)
frequency_btn.grid(row=2, column=2, padx=5, pady=5, sticky="nsew")



# Frame for additional tools
tools_frame = Frame(root, relief=RIDGE, bd=2)
tools_frame.grid(row=7, column=0, columnspan=3, pady=10, padx=20, sticky="ew")
tools_frame.rowconfigure(0, weight=1)
tools_frame.columnconfigure(0, weight=1)
tools_frame.columnconfigure(1, weight=1)
tools_frame.columnconfigure(2, weight=1)

# Button to run the Image pixel Analysis algorithm
image_stegnography = ttk.Button(tools_frame, text="图像提取", style="my.TButton", command=image_decode)
image_stegnography.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# Button to run the String Extraction Analysis algorithm
String_analysis = ttk.Button(tools_frame, text="字符串提取", style="my.TButton", command=string_analysis)
String_analysis.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

# Button to compare images
compare_button = ttk.Button(tools_frame, text="图像对比", style="my.TButton", command=compare_images)
compare_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

# Button to exit the program
style = ttk.Style()
style.configure('W.TButton', font = ('calibri', 10, 'bold'),foreground = 'red')

quitButton = ttk.Button(root, text="退出程序", style = 'W.TButton', command=root.quit)
quitButton.grid(row=8, column=2, pady=10, padx=20, sticky="e")

# Open the GUI
root.mainloop()