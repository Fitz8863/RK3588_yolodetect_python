import sys
import os
# 添加yoloFun所在的父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from yoloFun.func_yolov8_10 import myFunc # 推理引擎函数
import time
import cv2
from rknnlite.api import RKNNLite
from config import RKNN_MODEL, WIDTH, HEIGHT,FPS
from config import IMG_SIZE


if __name__ == '__main__':
    rknn = RKNNLite()

    print('--> Load RKNN model')
    rknn_init = rknn.load_rknn(RKNN_MODEL)
    if rknn_init != 0:
        print('Load RKNN model failed')
        exit(-1)
    print('done')

    # 这里选择使用npu的编号
    rknn_ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # 打开摄像头
    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture(
    f'v4l2src device=/dev/video0 ! image/jpeg, width={WIDTH}, height={HEIGHT}, framerate={FPS}/1 ! jpegdec ! videoconvert ! appsink',
    cv2.CAP_GSTREAMER)

    # 打开视频
    # capture = cv2.VideoCapture('video/spiderman.mp4')


    count = 0.0
    t1 = time.time()
    while True:
        count = count + 1

	    # 读取每一帧
        ret, frame = capture.read()
        if not ret:
            break
        
        # 预处理
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 这里执行推理
        re_img,packet = myFunc(rknn,frame)

        # 打印识别的坐标和类别 [x,y,type] -->  [['529.38', '206.09', 'red'], ['524.36', '411.55', 'blue']]
        print(packet)

        # 重新缩放回去
        re_img = cv2.resize(re_img, (WIDTH, HEIGHT))

        if count >= 60:
            fps = count / (time.time() - t1)
            t1 = time.time()
            count = 0.0
            print("fps= %.2f" % (fps))

        cv2.imshow("video", re_img)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
    #
    capture.release()
    cv2.destroyAllWindows()
    rknn.release()
