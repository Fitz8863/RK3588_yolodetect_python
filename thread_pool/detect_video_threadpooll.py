import sys
import os
# 添加yoloFun所在的父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import time
from rknnpool import rknnPoolExecutor #线程池
from yoloFun.func_yolov8_10 import myFunc # 推理引擎函数
from config import RKNN_MODEL,WIDTH, HEIGHT,FPS
from config import IMG_SIZE


# 线程池的初始化线程数
TPEs = 3

# 创建初始化线程池对象
pool = rknnPoolExecutor(
rknnModel=RKNN_MODEL,
TPEs=TPEs,
func=myFunc)

                              
if __name__=='__main__':

    # 打开摄像头
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(
    f'v4l2src device=/dev/video0 ! image/jpeg, width={WIDTH}, height={HEIGHT}, framerate={FPS}/1 ! jpegdec ! videoconvert ! appsink',
    cv2.CAP_GSTREAMER)

    # 打开视频
    # cap = cv2.VideoCapture('/home/orangepi/projects/rknn-python-rga-yolov5/videos/1k.mp4')

    # 线程池预处理
    if cap.isOpened():
        for i in range(TPEs+1):
            ret,frame = cap.read()
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                cap.release()
                del pool
                print("摄像头无法打开读取")
                exit(-1)
                # 预处理
            pool.put(frame)


    count = 0.0
    t1 = time.time()
    
    while cap.isOpened():
        count = count+1
        ret, frame = cap.read()
        if not ret:
            cap.release()
            pool.release()
            print("摄像头无法打开读取")
            exit(-1)
            
        # 预处理
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # 线程池处理
        pool.put(frame)
        [re_img, packet], flag = pool.get()
        if not flag:
            print("---------推理过程出现错误------------")
            break
        
        # 打印识别的坐标和类别 [x,y,type] -->  [['529.38', '206.09', 'red'], ['524.36', '411.55', 'blue']]
        # print(packet)

        # 重新缩放回去
        re_img = cv2.resize(re_img, (WIDTH, HEIGHT))

        cv2.imshow("video", re_img)
        
        if (cv2.waitKey(1) & 0xff) == 27:
            break

        # 计算帧率并且打印
        if count >= 60:
            fps = count / (time.time() - t1)
            t1 = time.time()
            count = 0.0
            print("fps= %.2f" % (fps))
        
    cap.release()
    pool.release()
