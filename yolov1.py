from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, BatchNormalization, Input, \
    LeakyReLU, Reshape, Flatten


def LeakyConvBlock(input, filter_size, kernel_size, strides=(1, 1)):
    x = Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding='same', activation=None)(input)
    return LeakyReLU(alpha=0.1)(x)


def LeakyMaxpoolBlock(input):
    x = MaxPool2D()(input)
    return LeakyReLU(alpha=0.1)(x)


def QuadroLeakyConvBlock(input):
    x = LeakyConvBlock(input, filter_size=256, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=512, kernel_size=3)
    x = LeakyConvBlock(x, filter_size=256, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=512, kernel_size=3)
    x = LeakyConvBlock(x, filter_size=256, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=512, kernel_size=3)
    x = LeakyConvBlock(x, filter_size=256, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=512, kernel_size=3)
    return x


def DualLeakyConvBlock(input):
    x = LeakyConvBlock(input, filter_size=512, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=1024, kernel_size=3)
    x = LeakyConvBlock(x, filter_size=512, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=1024, kernel_size=3)
    return x


INPUT_LAYER = Input(shape=(448, 448, 3))

def Yolov1Model():
    inputs = INPUT_LAYER

    # 1번째
    x = LeakyConvBlock(inputs, filter_size=64, kernel_size=7, strides=2)
    # x = LeakyMaxpoolBlock(x)  # MaxPool 이후에는 LRELU를 사용하지 않는다?
    x = MaxPool2D()(x)

    # 2번째
    x = LeakyConvBlock(x, filter_size=192, kernel_size=3)
    # x = LeakyMaxpoolBlock(x)
    x = MaxPool2D()(x)

    # 3번째
    x = LeakyConvBlock(x, filter_size=128, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=256, kernel_size=3)
    x = LeakyConvBlock(x, filter_size=256, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=512, kernel_size=3)
    # x = LeakyMaxpoolBlock(x)
    x = MaxPool2D()(x)

    # 4번째
    x = QuadroLeakyConvBlock(x)
    x = LeakyConvBlock(x, filter_size=512, kernel_size=1)
    x = LeakyConvBlock(x, filter_size=1024, kernel_size=3)
    # x = LeakyMaxpoolBlock(x)
    x = MaxPool2D()(x)

    # 5번째
    x = DualLeakyConvBlock(x)
    x = LeakyConvBlock(x, filter_size=1024, kernel_size=3)
    x = LeakyConvBlock(x, filter_size=1024, kernel_size=3, strides=2)

    # 6번째
    x = LeakyConvBlock(x, filter_size=1024, kernel_size=3)
    x = LeakyConvBlock(x, filter_size=1024, kernel_size=3)

    # 7번째
    x = Flatten()(x)
    x = Dense(4096)(x)
    # x = LeakyReLU(alpha=0.1)(x) # Dense 레이어에소드 LeakyRELU를 사용하지 않는다?

    # 8번째 (Output)
    x = Dense(1470)(x)
    # x = LeakyReLU(alpha=0.1)(x)
    x = Reshape((7, 7, 30))(x)

    model = Model(inputs, x)
    return model


import tensorflow.keras.backend as K


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


# Expanded ? * 7 * 7 * 1 * 4
def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last, result = 7 * 7

    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]]) # 늘어놓는 함수  tile -> 같은걸 N번 반복함
    # 결과 -> 0~6, 0~6, ...., 0~6

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1]) # tile을 [n, m] 쓰면 dims 2로 만들어줌
    # 결과 -> [0~6], [0~6], [0~6], ...

    conv_width_index = K.flatten(K.transpose(conv_width_index))
    # 결과 -> 0, 0, 0, 0, 0, 0, 0 (7개), 1, 1, 1, 1, 1, 1, 1 (7개), ...

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    # 결과 -> [0, 0], [1, 0], [2, 0], ..., [5, 6], [6, 6]

    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    # 결과 -> 1 * 7 * 7 에 있는 [1 * 2]의 conv index item이 만들어짐
    # 각각 [1 * 2]의 값은 [0, 0], [1, 0], [2, 0], ..., [5, 6], [6, 6]
    # 이런 식으로 이루어져 있음 -> Mask를 만들기 위한 과정
    # 결과 shape -> 1, 7, 7, 1, 2

    conv_index = K.cast(conv_index, K.dtype(feats))
    # 타입 맞추기
    # 마지막 box_xy, box_wh에서 덧셈/나눗셈 연산 위해 타입 맞추기

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))
    # [7, 7]을 앞자리 [1, 1, 1]로 맞추기
    # [1, 2]로 맞추어 진행
    # 결과 shape -> 1, 1, 1, 1, 2

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    # [..., :2]의 결과 -> [1, 7, 7, 1, 2] (본래 뒷자리 [1, 4]였는데, 앞에 2개를 사용)
    box_wh = feats[..., 2:4] * 448
    # [..., 2:4]의 결과 -> [1, 7, 7, 1, 2] (본래 뒷자리 [1, 4]였는데, 뒤에 2개를 사용)

    return box_xy, box_wh


'''
    y_true.shape = [7, 7, 25]
    0  ~ 19 (20) -> one-hot class
    20 ~ 23 (4)  -> [x, y, w, h]
    24      (1)  -> response???? responsible mask!
'''

'''
    y_pred.shape = [?, 7, 7, 30]
    0  ~ 19 (20) -> predicted class probability
    20 ~ 21 (2)  -> predicted trust values (CONFIDENCE!!!)
    22 ~ 29 (8)  -> predicted bounding boxes [x, y, w, h], [x, y, w, h]
'''
def Yolov1Loss(y_true, y_pred):
    label_class = y_true[..., :20]      # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]      # ? * 7 * 7 * 4
    responsible_mask = y_true[..., 24]  # ? * 7 * 7
    responsible_mask = K.expand_dims(responsible_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_bbox_confidences = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])  # ? * 7 * 7 * 1 * 4 (4 -> 1 * 4)
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])  # ? * 7 * 7 * 2 * 4 (8 -> 2 * 4)

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    # Loss 함수 4번 (with lambda_noobj 0.5)
    no_object_loss = 0.5 * (1 - box_mask * responsible_mask) * K.square(0 - predict_bbox_confidences)
    # Loss 함수 3번 (without lambda_noobj)
    object_loss = box_mask * responsible_mask * K.square(1 - predict_bbox_confidences)

    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    # Loss 함수 5번
    class_loss = responsible_mask * K.square(label_class - predict_class)

    # Loss 함수 5번 총합
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    responsible_mask = K.expand_dims(responsible_mask)

    # Loss 함수 1번
    box_loss = 5 * box_mask * responsible_mask * K.square((label_xy - predict_xy) / 448)

    # Loss 함수 2번
    box_loss += 5 * box_mask * responsible_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)

    # 1번+2번 총합
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss