import sys
import os
# 添加yoloFun所在的父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from yoloFun.func_yolov8_10 import myFunc # 推理引擎函数
import time
import cv2
from rknnlite.api import RKNNLite
from config import RKNN_MODEL
from config import IMG_SIZE

if __name__ == '__main__':
    # 读取图片
    img = cv2.imread('data/bus.jpg')

    # 获取图片原始分辨率
    height, width = img.shape[:2]

    rknn = RKNNLite()

    print('--> Load RKNN model')
    rknn_init = rknn.load_rknn(RKNN_MODEL)
    if rknn_init != 0:
        print('Load RKNN model failed')
        exit(-1)
    print('done')

    # 这里选择使用npu的编号
    rknn_ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    
    # 预处理
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 这里执行推理
    re_img,packet = myFunc(rknn,img)

    # 打印识别的坐标和类别 [x,y,type] -->  [['529.38', '206.09', 'red'], ['524.36', '411.55', 'blue']]
    print(packet)

    # 重新缩放回去
    re_img = cv2.resize(re_img, (width, height))

    # 保存
    cv2.imwrite('result.jpg', re_img)

