import cv2
import numpy as np
from config import OBJ_THRESH, NMS_THRESH, IMG_SIZE, CLASSES


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2]) * 2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE / grid_h)

    box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

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


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[12, 16], [19, 36], [40, 28],
               [36, 75], [75, 55], [72, 146],
               [142, 110], [192, 243], [459, 401]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    data = []
    for box, score, cl in zip(boxes, scores, classes):
        left, top, right, bottom = box

        # 转换为整数坐标
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        # 计算中心点（用于记录）
        X = (left + right) / 2
        Y = (top + bottom) / 2

        target = CLASSES[cl]

        # --- 画矩形框 ---
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

        # --- 画中心点 ---
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)  # 绿色圆点

        # --- 绘制文本标签 ---
        cv2.putText(image, f'{target} {score:.2f}',
                    (left, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # --- 保存输出数据 ---
        data.append([
            str(format(X * (1280 / 640), '.2f')),
            str(format(Y * (720 / 640), '.2f')),
            target
        ])

    return data


    # if cl == 0:
    #     cv2.putText(image, coordinates_text,
    #     (5, 10),  # 将文本放置在左上角
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5, (255, 0, 0), 2)
    # elif cl == 1:
    #     cv2.putText(image, coordinates_text,
    #     (5, 30),  # 将文本放置在左上角
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5, (0, 0, 255), 2)
    # elif cl == 2:
    #     cv2.putText(image, coordinates_text,
    #     (5, 50),  # 将文本放置在左上角
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5, (0, 255, 0), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
             new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    # return im
    return im, ratio, (dw, dh)


def myFunc(rknn_lite, IMG):
    # # 等比例缩放
    # IMG, ratio, (dw, dh) = letterbox(IMG)
    # # 强制放缩
    # IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))
    # IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

    packet = []

    IMG = np.expand_dims(IMG, 0)

    outputs = rknn_lite.inference(inputs=[IMG])

    input0_data = outputs[0].reshape([3, -1] + list(outputs[0].shape[-2:]))
    input1_data = outputs[1].reshape([3, -1] + list(outputs[1].shape[-2:]))
    input2_data = outputs[2].reshape([3, -1] + list(outputs[2].shape[-2:]))

    input_data = []
    for i in range(3):
        output = outputs[i]
        bs, c, h, w = output.shape
        output = output.reshape((bs, 3, 85, h, w))   # 拆分3个anchor
        output = np.transpose(output, (0, 3, 4, 1, 2))  # -> (bs, h, w, 3, 85)
        input_data.append(output[0])  # 去掉batch维

    boxes, classes, scores = yolov5_post_process(input_data)

    IMG = cv2.cvtColor(IMG[0], cv2.COLOR_RGB2BGR)
    if boxes is not None:
        packet = draw(IMG, boxes, scores, classes)

    return [IMG, packet]
