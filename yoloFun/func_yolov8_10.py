import cv2
import numpy as np
from config import OBJ_THRESH, NMS_THRESH, IMG_SIZE, CLASSES

MODEL_SIZE = (IMG_SIZE, IMG_SIZE)

color_palette = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def letter_box(im, new_shape, pad_color=(0, 0, 0), info_need=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border

    if info_need is True:
        return im, ratio, (dw, dh)
    else:
        return im


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def dfl(position):
    # Distribution Focal Loss (DFL)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([MODEL_SIZE[1] // grid_h, MODEL_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw_detections(img, left, top, right, bottom, score, class_id):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.
    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.
    Returns:
        None
    """

    # Retrieve the color for the class ID
    color = color_palette[class_id]

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), color, 2)

    # 计算矩形框的中心点坐标
    center_point = (int((left + right) / 2), int((top + bottom) / 2))

    # 在中心点画一个标记点
    cv2.circle(img, center_point, 5, (0, 0, 255), -1)

    # Create the label text with class name and score
    label = f"{CLASSES[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Calculate the position of the label text
    label_x = left
    label_y = top - 10 if top - 10 > label_height else top + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                  cv2.FILLED)

    # Draw the label text on the image
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw(image, boxes, scores, classes):
    data = []
    img_h, img_w = image.shape[:2]
    # Calculate scaling factors for bounding box coordinates
    if (img_w >= img_h):
        x_factor = img_w / MODEL_SIZE[0]
        y_rsz = img_h * MODEL_SIZE[1] / img_w
        y_factor = img_h / y_rsz
        offset = (MODEL_SIZE[1] - y_rsz) / 2
    else:
        y_factor = img_h / MODEL_SIZE[1]
        x_rsz = img_w * MODEL_SIZE[0] / img_h
        x_factor = img_w / x_rsz
        offset = (MODEL_SIZE[0] - x_rsz) / 2

    for box, score, cl in zip(boxes, scores, classes):

        x1, y1, x2, y2 = [int(_b) for _b in box]

        if (img_w >= img_h):
            left = int(x1 * x_factor)
            top = int((y1 - offset) * y_factor)
            right = int(x2 * x_factor)
            bottom = int((y2 - offset) * y_factor)
        else:
            left = int((x1 - offset) * x_factor)
            top = int(y1 * y_factor)
            right = int((x2 - offset) * x_factor)
            bottom = int(y2 * y_factor)
        X = (x1 + x2) / 2
        Y = (y1 + y2) / 2
        target = CLASSES[cl]
        data.append([str(format(X * (1280 / 640), '.2f')), str(format(Y * (720 / 640), '.2f')), target])
        draw_detections(image, left, top, right, bottom, score, cl)
    return data


def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换
    Returns:指定尺寸的图像
    """
    from PIL import Image
    ih, iw, _ = image.shape
    h, w = size
    if letterbox_image:
        scale = min(w / iw, h / ih)  # 缩放比例
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # 生成画布
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        image_back = image
    return image_back


def myFunc(rknn_lite, IMG):
    
    # # 等比例缩放
    # IMG, ratio, (dw, dh) = letterbox(IMG)
    # # 强制放缩
    # IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))
    # IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    
    packet = []

    input = np.expand_dims(IMG, 0)

    # outputs = rknn_lite.inference(inputs=[IMG]) print

    outputs = rknn_lite.inference([input])

    boxes, classes, scores = post_process(outputs)

    IMG = cv2.cvtColor(IMG, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        packet = draw(IMG, boxes, scores, classes)

    # IMG = cv2.cvtColor(IMG[0], cv2.COLOR_RGB2BGR)
    # if boxes is not None:
    #     packet = draw(IMG, boxes, scores, classes)

    return [IMG, packet]