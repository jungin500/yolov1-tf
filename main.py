# INFO까지의 로그 Suppress하기
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from yolov1 import Yolov1Model, Yolov1Loss, INPUT_LAYER
from DataGenerator import Dataloader


def convert_yolobbox_tfbbox(yolo_bbox, image_size=448):
    """
        yolo_bbox
        - Shape: [1 * 7 * 7 * 4]
        - [x_c, y_c, w, h]

        - Output> [1 * 7 * 7 * 4] and [y1, x1, y2, x2]
    """
    return np.stack([
        (yolo_bbox[:, :, :, 1] - yolo_bbox[:, :, :, 3] / 2) * image_size,
        (yolo_bbox[:, :, :, 0] - yolo_bbox[:, :, :, 2] / 2) * image_size,
        (yolo_bbox[:, :, :, 1] + yolo_bbox[:, :, :, 3] / 2) * image_size,
        (yolo_bbox[:, :, :, 0] + yolo_bbox[:, :, :, 2] / 2) * image_size
    ], axis=3)


# result: 1 * 7 * 7 * 30
def postprocess_non_nms_result(input_image, network_output):
    classes = network_output[:, :, :, :20]
    confidence_1, confidence_2 = network_output[:, :, :, 20], network_output[:, :, :, 21]
    bbox_1, bbox_2 = network_output[:, :, :, 22:26], network_output[:, :, :, 26:30]

    class_score_bbox_1 = np.expand_dims(confidence_1, axis=3) * classes
    class_score_bbox_2 = np.expand_dims(confidence_2, axis=3) * classes

    # Set zero if score < thresh1 (0.2)
    class_score_bbox_1[np.where(class_score_bbox_1 < thresh1)] = 0.
    class_score_bbox_2[np.where(class_score_bbox_2 < thresh1)] = 0.

    # class_score 중에서 가장 높은 class id
    class_score_bbox_1_max_class = np.argmax(class_score_bbox_1, axis=3)
    class_score_bbox_2_max_class = np.argmax(class_score_bbox_2, axis=3)
    class_score_bbox_1_max_score = np.amax(class_score_bbox_1, axis=3)
    class_score_bbox_2_max_score = np.amax(class_score_bbox_2, axis=3)

    batch_size = np.shape(input_image)[0]
    for batch in range(batch_size):
        input_image_single = input_image[batch]
        input_image_pil = Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')
        input_image_draw = ImageDraw.Draw(input_image_pil)

        for y in range(7):
            for x in range(7):
                first_bigger = class_score_bbox_1_max_score[batch][y][x] > class_score_bbox_2_max_score[batch][y][x]
                if (first_bigger and class_score_bbox_1_max_score[batch][y][x] == 0) and (
                        not first_bigger and class_score_bbox_2_max_score[batch][y][x] == 0):
                    continue

                class_id = None
                class_score_bbox = None
                bbox = None
                if first_bigger:
                    class_id = class_score_bbox_1_max_class[batch][y][x]
                    class_score_bbox = class_score_bbox_1_max_score[batch][y][x]
                    bbox = bbox_1[batch][y][x]
                else:
                    class_id = class_score_bbox_2_max_class[batch][y][x]
                    class_score_bbox = class_score_bbox_2_max_score[batch][y][x]
                    bbox = bbox_2[batch][y][x]

                if class_score_bbox < thresh2:
                    continue

                (x_c, y_c, w, h) = bbox
                x_1 = (x_c - (w / 2)) * 448
                y_1 = (y_c - (h / 2)) * 448
                x_2 = (x_c + (w / 2)) * 448
                y_2 = (y_c + (h / 2)) * 448

                input_image_draw.rectangle([x_1, y_1, x_2, y_2], outline="red", width=3)
                input_image_draw.text([x_1 + 5, y_1 + 5], text=str(class_score_bbox), fill='yellow')
                input_image_draw.text([x_1 + 5, y_1 + 13], text=train_data.GetLabelName(class_id), fill='yellow')

        input_image_pil.show(title="Sample Image")


# result: 1 * 7 * 7 * 30
def postprocess_result(input_image, network_output):
    classes = network_output[:, :, :, :20]
    confidence_1, confidence_2 = network_output[:, :, :, 20], network_output[:, :, :, 21]
    bbox_1, bbox_2 = network_output[:, :, :, 22:26], network_output[:, :, :, 26:30]

    class_score_bbox_1 = np.expand_dims(confidence_1, axis=3) * classes
    class_score_bbox_2 = np.expand_dims(confidence_2, axis=3) * classes
    class_scores = np.stack([class_score_bbox_1, class_score_bbox_2], axis=3)  # 1 * 7 * 7 * 2 * 20

    # 문제없이 작동하는 부분
    class_scores = np.reshape(class_scores, [-1, 20])  # 98(7 * 7 * 2) * 20

    # Set zero if score < thresh1 (0.2)
    class_scores[np.where(class_scores < thresh1)] = 0.

    return  # invalid return here!

    # for class_id in range(np.shape(class_scores)[1]):
    #     # Sort in Descending order - intermediate value, not a reference
    #     k = class_scores[class_scores[:, 0].argsort()[::-1], 0]

    # NMS: check bbox_max

    # Sort descending by its scores

    # apply NMS with Tensorflow...
    # tf_nms_boxes1, tf_nms_boxes2 = convert_yolobbox_tfbbox(bbox_1), convert_yolobbox_tfbbox(bbox_2)
    # tf_nms_boxes = np.stack([tf_nms_boxes1, tf_nms_boxes2], axis=3)
    # tf_nms_boxes = np.reshape(tf_nms_boxes, [-1, 4])  # 98 * 4
    #
    # bbox_to_draw = np.empty(shape=(1, 7, 7, 2, 4))
    # for class_id in range(np.shape(class_scores)[1]):
    #     # class_id는 0부터 19까지 (클래스 개수만큼) 진행
    #     single_class_score = class_scores[:, class_id]
    #
    #     selected_indices = tf.image.non_max_suppression(
    #         boxes=tf_nms_boxes,
    #         scores=single_class_score,
    #         iou_threshold=0.5,
    #         max_output_size=tf.convert_to_tensor(2)
    #     )
    #
    #     print(selected_indices)
    #     print(selected_indices.shape)
    #
    #     nms_selected = tf.gather(single_class_score, selected_indices)

# train_data = Dataloader(file_name='manifest_train.txt', numClass=20, batch_size=4)
# valid_train_data = Dataloader(file_name='manifest_valid.txt', numClass=20, batch_size=4)
# test_data = Dataloader(file_name='manifest_test.txt', numClass=20, batch_size=4)
dev_data = Dataloader(file_name='manifest_two.txt', numClass=20, batch_size=2, augmentation=True)
train_data = dev_data
valid_train_data = dev_data
test_data = dev_data

model = Yolov1Model()
optimizer = Adam(learning_rate=1e-5, decay=5e-4)

model.compile(optimizer=optimizer, loss=Yolov1Loss)

# model.summary()
# plot_model(model, to_file='model.png')
# model_image = cv2.imread('model.png')
# cv2.imshow("image", model_image)
# cv2.waitKey(0)

GLOBAL_EPOCHS = 200
SAVE_PERIOD = GLOBAL_EPOCHS / 2
CHECKPOINT_FILENAME = 'checkpoint.h5'
MODE_TRAIN = False
LOAD_WEIGHT = True

thresh1 = 0.2
thresh2 = 0.2


save_best_model = ModelCheckpoint(
    CHECKPOINT_FILENAME,
    save_best_only=True,
    save_weights_only=True,
    save_freq='epoch',
    period=SAVE_PERIOD
)

if LOAD_WEIGHT:
    model.load_weights(CHECKPOINT_FILENAME)

if MODE_TRAIN:
    model.fit(
        train_data,
        epochs=GLOBAL_EPOCHS,
        validation_data=train_data,
        shuffle=False,
        callbacks=[save_best_model]
    )
else:
    import random

    data_iterations = 1
    result_set = []
    for _ in range(data_iterations):
        image, _ = train_data.__getitem__(random.randrange(0, train_data.__len__()))
        result = model.predict(image)
        postprocess_non_nms_result(image, result)

    print(result_set)
