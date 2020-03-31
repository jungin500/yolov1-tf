# INFO까지의 로그 Suppress하기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from yolov1 import Yolov1Model, INPUT_LAYER
from DataGenerator import Dataloader
import numpy as np

LOSS_COORD = 5
LOSS_NOOBJ = .5
GRID_SIZE = 7


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def create_yolo_loss(input_layer):
    def draw_yolov1_loss_result(y_pred):
        pass


    # Assume that y_pred_bbox.shape = ? * 7 * 7 * 5
    # Returns: two (? * 7 * 7 * 1 * 2)
    def convert_pred_to_minmax(y_pred_bbox):
        (confidence, x_center, y_center, width, height) = y_pred_bbox
        x_min = x_center - (width / 2) * 448
        x_max = x_center + (width / 2) * 448
        y_min = y_center - (height / 2) * 448
        y_max = y_center + (height / 2) * 448
        return tf.stack([x_min, y_min], axis=3), tf.stack([x_max, y_max], axis=3)

    # y_pred: [B, 7, 7, 30]
    # Eager Execution
    # (Graph Execetion보다 속도가 느리지만 Iteration이 가능하다.
    # @tf.function
    def yolo_loss(y_true, y_pred):
        '''
        YOLOv1 Loss Function

        :param y_true: 정답값 (7*7*6)
        :param y_pred: 예측값 (7*7*30)
        :return: Loss의 결과
        '''

        # draw_yolov1_loss_result(y_pred)

        # y_pred: [batch, 7, 7, 30]
        # batch_size, cell_shape_x, cell_shape_y, cell_shape_infos = y_true.shape
        # if cell_shape_x != cell_shape_y:
        #     raise RuntimeError('Cell shape is not same: {} != {}'.format(cell_shape_x, cell_shape_y))

        # center_x = y_true[:, :, 1]
        # center_y = y_true[:, :, 2]
        # width = y_true[:, :, 3]
        # height = y_true[:, :, 4]
        # class_id = y_true[:, :, 5]

        '''
        pow_of_difference = [ c_diff_pow, x_diff_pow, y_diff_pow, w_diff_pow, h_diff_pow ]
        '''
        y_pred_bbox_1 = y_pred[:, :, :, :5]       # [B: 7: 7: 5]
        y_pred_bbox_2 = y_pred[:, :, :, 5:10]     # [B: 7: 7: 5]
        y_pred_class_prob = y_pred[:, :, :, 10:]  # [B: 7: 7:20]
        y_pred_xy_min, y_pred_xy_max = convert_pred_to_minmax(y_pred_bbox_1) # ? * 7 * 7 * 1 * 2

        y_true_bbox   = y_true[:, :, :, :5]       # [B: 7: 7: 5] -> [c, x, y, w, h]
        y_true_class  = y_true[:, :, :, 5]        # [B: 7: 7] -> Class ID
        y_true_xy_min, y_true_xy_max = convert_true_to_minmax(y_true_bbox)

        best_iou_mask =

        # Interpret as (1, 7, 7)
        obj_responsible = best_iou_mask * tf.expand_dims(y_true[:, :, :, 0])
        # obj_nonresponsible = 1 - y_true[:, :, :, 0] # 1을 0으로, 0을 1로

        # 계산 결과들
        pow_of_difference_1 = tf.pow(y_pred_bbox_1 - y_true_bbox, 2)
        pow_of_difference_2 = tf.pow(y_pred_bbox_2 - y_true_bbox, 2)

        # ReLU는 0 미만의 값을 자르기 위해 사용 (sqrt 함수 들어갈 때).
        sum_of_squared_difference_1 = tf.pow(tf.sqrt(tf.nn.relu(y_pred_bbox_1)) - tf.sqrt(tf.nn.relu(y_true_bbox)), 2)
        sum_of_squared_difference_2 = tf.pow(tf.sqrt(tf.nn.relu(y_pred_bbox_2)) - tf.sqrt(tf.nn.relu(y_true_bbox)), 2)

        # 1번째 수식
        ssd_1st = LOSS_COORD * obj_responsible * (
            pow_of_difference_1[:, :, :, 1] + pow_of_difference_1[:, :, :, 2] +
            pow_of_difference_2[:, :, :, 1] + pow_of_difference_2[:, :, :, 2]
        )

        # 2번째 수식
        ssd_2nd = LOSS_COORD * obj_responsible * (
            sum_of_squared_difference_1[:, :, :, 3] + sum_of_squared_difference_1[:, :, :, 4] +
            sum_of_squared_difference_2[:, :, :, 3] + sum_of_squared_difference_2[:, :, :, 4]
        )

        # 3번째 수식 - 모든 Confidence의 SSD 합해서 더하기
        ssd_3rd = obj_responsible * (pow_of_difference_1[:, :, :, 0] + pow_of_difference_2[:, :, :, 0])

        # 4번째 수식 - Non-Responsible한 SSD에 가중치 합해서 더하기
        ssd_4th = LOSS_NOOBJ * obj_nonresponsible * (
            pow_of_difference_1[:, :, :, 0] + pow_of_difference_2[:, :, :, 0]
        )
        
        # 5번째 수식 - 각 셀마다 나타나는 모든 클래스 확률의 차이를 합해서 더하기
        ssd_5th = tf.reduce_sum(
            obj_responsible * tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(y_true_class, dtype=tf.dtypes.int32),
                logits=y_pred_class_prob
            )
        )

        # dummy result loss
        # return tf.reduce_sum(y_pred)
        return tf.reduce_sum(ssd_1st + ssd_2nd + ssd_3rd + ssd_4th) + ssd_5th

    def dummy_loss(y_true, y_pred):
        return tf.reduce_sum(y_pred)

    return yolo_loss


train_data = Dataloader(file_name='train_list.txt', numClass=20)

model = Yolov1Model()
optimizer = Adam(learning_rate=1e-3, decay=5e-4)

# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['binary_accuracy'])
model.compile(optimizer=optimizer, loss=create_yolo_loss(INPUT_LAYER))

model.summary()
# model.fit(train_data, epochs=20, validation_data=None, shuffle=False)
