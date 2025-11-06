# RK3588_yolodetect

这里提供yolov5，yolov7，yolov8(ultralytics)的推理python代码，包括单线程和线程池版本

## 环境说明

### PC端

Python版本：conda环境，Python 3.10.16

rknn-toolkit2版本：Version: 2.0.0b0+9bab5682

```bash
# 可以通过这个指令查看
pip3 show rknn-toolkit2
```



### 香橙派rk3588端

Python版本：Python 3.10.12

NPU版本：RKNPU driver: v0.9.6

```bash
# 可通过下面这个查看
sudo cat /sys/kernel/debug/rknpu/version
```

rknn-toolkit-lite2：Version: 1.6.0

```bash
# 可以通过下面这个指令查看
pip3 show rknn-toolkit-lite2
```

## 项目文件结构

拉取仓库代码：

```bash
git clone https://github.com/Fitz8863/RK3588_yolodetect.git
```

然后进入目录

```bash
cd RK3588_yolodetect
```

### 代码文件结构

![image-20251106214318279](README_images\image-20251106214318279.png)

这里可以看到有两个文件夹，其中signal_threaded 这个目录下是表示使用单线程去推理；thread_pool 是使用线程池去推理

- detect_img.py：表示单线程下，推理图片的执行入口代码
- detect_video.py：表示单线程下，视频推理的执行入口代码
- detect_video_threadpooll.py：表示线程池下，视频推理的执行入口代码

![image-20251106220737886](README_images\image-20251106220737886.png)

在yoloFun文件夹可以看到有三个文件

- func_yolov5.py：表示这个是yolov5推理使用的到的推理文件
- func_yolov7.py：表示这个是yolov7推理使用的到的推理文件
- yolo_func0_10.py：表示这个是yolov8~yolov10推理使用的到的推理文件

在当前目录下有一个config.py文件，这个是基本参数的配置文件

### 模型结构

模型文件在rknn_models目录下，请你量化的时候按照下面这些输入输出的格式去量化

1.下面是yolov5s.rknn的输入输出结构

![image-20251106221633375](README_images\image-20251106221633375.png)

2.下面是yolov7.rknn的输入输出结构

![image-20251106221728878](README_images\image-20251106221728878.png)

3.下面是yolov8s.rknn的输入输出结构

![image-20251106221803656](README_images\image-20251106221803656.png)

## 运行案例

比如要运行yolov8的模型，首先去看到config.py文件，这里选择RKNN的模型

![image-20251106223155641](README_images\image-20251106223155641.png)

然后去到你要执行的那个入口文件，比如detect_img.py ,然后修改这个推理函数的导入，如果是yolov8，那就导入yolov8_10（如果要推理yolov5就改成yolov5的就行了）

![image-20251106223309735](README_images\image-20251106223309735.png)

执行下面这个语句

```bash
python single_threaded/detect_img.py 
```

可以看到生成一个result.jpg的图片，这个是推理后的结果

如果要推理视频的话就执行下面这个，另外视频输入源可以在detect_video.py 里面去修改，比如输入是摄像头视频或者mp4视频

```
python single_threaded/detect_video.py 
```



