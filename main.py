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

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from yolo_loss_gh import model_tiny_yolov1


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

import random

# result: 1 * 7 * 7 * 30
def postprocess_non_nms_result(input_image, network_output, no_suppress=False, display_all=True):
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
        if not display_all:
            if random.randrange(2) == 1:
                continue

        input_image_single = input_image[batch]
        input_image_pil = Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')
        input_image_draw = ImageDraw.Draw(input_image_pil)

        for y in range(7):
            for x in range(7):
                first_bigger = class_score_bbox_1_max_score[batch][y][x] > class_score_bbox_2_max_score[batch][y][x]
                if not no_suppress and (first_bigger and class_score_bbox_1_max_score[batch][y][x] == 0) and (
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

                if not no_suppress and class_score_bbox < thresh2:
                    continue

                (x_c, y_c, w, h) = bbox

                x_c, y_c = [x_c + x, y_c + y]
                x_c, y_c = [x_c / 7, y_c / 7]

                x_1 = (x_c - (w / 2)) * 448
                y_1 = (y_c - (h / 2)) * 448
                x_2 = (x_c + (w / 2)) * 448
                y_2 = (y_c + (h / 2)) * 448

                input_image_draw.rectangle([x_1, y_1, x_2, y_2], outline="red", width=3)
                input_image_draw.text([x_1 + 5, y_1 + 5], text=str(class_score_bbox), fill='yellow')
                input_image_draw.text([x_1 + 5, y_1 + 13], text=train_data.GetLabelName(class_id), fill='yellow')

        input_image_pil.show(title="Sample Image")


def create_index_map(begin, end=None, shape=(1, 7, 7, 1)):
    base_range = tf.range(begin, end)
    reshaped_tiles = tf.reshape(base_range, shape)
    return tf.cast(reshaped_tiles, tf.float32)


def yolov1_precision(y_true, y_pred):
    label_class = y_true[..., :20]  # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]  # ? * 7 * 7 * 4
    responsible_mask = y_true[..., 24]  # ? * 7 * 7

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_bbox_confidences = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    predict_class_probabilities_bbox_1 = predict_class * predict_bbox_confidences[..., 0:1]
    predict_class_probabilities_bbox_2 = predict_class * predict_bbox_confidences[..., 1:2]

    # Interleave two probs in memory
    predict_class_probabilities = tf.reshape(
        tf.stack(
            [predict_class_probabilities_bbox_1, predict_class_probabilities_bbox_2],
            axis=3
        ),
        (-1, 98, 20)
    )

    # Set zero if score < thresh1
    predict_class_probabilities = tf.where(
        predict_class_probabilities < thresh1,
        0,
        predict_class_probabilities
    )

    # Sort descending
    index_map = create_index_map(98, shape=(1, 98, 1))

    '''
        <tf.Tensor: shape=(1, 98, 21), dtype=float32, numpy=
        array([[[ 0.,  1.,  1., ...,  1.,  1.,  1.],
                [ 1.,  2.,  2., ...,  2.,  2.,  2.],
                [ 2.,  1.,  1., ...,  1.,  1.,  1.],
                ...,
                [95.,  2.,  2., ...,  2.,  2.,  2.],
                [96.,  1.,  1., ...,  1.,  1.,  1.],
                [97.,  2.,  2., ...,  2.,  2.,  2.]]], dtype=float32)>
    '''
    classprob_bbox_mapped = tf.concat([index_map, predict_class_probabilities], axis=2)

    # 0번째는 index이다...
    tf.argmax()

GLOBAL_EPOCHS = 4000
# SAVE_PERIOD_EPOCHS = 100
SAVE_PERIOD_SAMPLES = 200
# CHECKPOINT_FILENAME = 'checkpoint.h5'
# CHECKPOINT_FILENAME = "saved-model-a2ae-{epoch:02d}.hdf5"
CHECKPOINT_FILENAME = "yolov1-voc2007-technique.hdf5"
MODEL_SAVE = True
MODE_TRAIN = True
INTERACTIVE_TRAIN = True
LOAD_WEIGHT = False

train_data = Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=8, augmentation=True)
train_data_no_augmentation = Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=4, augmentation=False)
valid_train_data = Dataloader(file_name='manifest-valid.txt', numClass=20, batch_size=2)
test_data = Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=2)
train_two_data = Dataloader(file_name='manifest-two.txt', numClass=20, batch_size=2, augmentation=False)
dev_data = Dataloader(file_name='manifest-two.txt', numClass=20, batch_size=2, augmentation=False)
dev_32_data = Dataloader(file_name='manifest-eight.txt', numClass=20, batch_size=8, augmentation=True)

TARGET_TRAIN_DATA = train_two_data
# train_data = dev_data
valid_train_data = train_two_data
# test_data = dev_data


'''
    Learning Rate에 대한 고찰
    - 다양한 Augmentation이 활성화되어 있을 시, 2e-5  (loss: 100 언저리까지 가능)
    - Augmentation 비활성화 시, 1e-4: loss 20 언저리까지 가능
    - 1e-5: 20 언저리까지 떨어진 이후
    - Augmentation 비활성화 시, 시작부터 5e-6: 23까지는 잘 떨어짐
'''
LEARNING_RATE = 5e-6
DECAY_RATE = 5e-5
thresh1 = 0.2
thresh2 = 0.2

input_shape = (448, 448, 3)
inputs = Input(input_shape)
yolo_outputs = model_tiny_yolov1(inputs)
model = Model(inputs=inputs, outputs=yolo_outputs)

# model = Yolov1Model()
# optimizer = Adam(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
# model.compile(optimizer=optimizer, loss=Yolov1Loss) # , metrics=[yolov1_precision])

model.compile(optimizer='adam', loss=Yolov1Loss)
# model.summary()
# plot_model(model, to_file='model.png')
# model_image = cv2.imread('model.png')
# cv2.imshow("image", model_image)
# cv2.waitKey(0)

# save_frequency = int(
#     SAVE_PERIOD_EPOCHS * TARGET_TRAIN_DATA.__len__() / TARGET_TRAIN_DATA.batch_size *
#     (1 if TARGET_TRAIN_DATA.augmenter else TARGET_TRAIN_DATA.augmenter_size)
# )
save_frequency_raw = SAVE_PERIOD_SAMPLES
print("Save frequency is {} sample, batch_size={}.".format(save_frequency_raw, TARGET_TRAIN_DATA.batch_size))

save_best_model = ModelCheckpoint(
    CHECKPOINT_FILENAME,
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    # save_freq=save_frequency
    save_freq=save_frequency_raw
)

if LOAD_WEIGHT:
    model.load_weights(CHECKPOINT_FILENAME)


if MODE_TRAIN:
    if INTERACTIVE_TRAIN:
        import random

        epoch_divide_by = 5
        epoch_iteration = 0
        while epoch_iteration * (GLOBAL_EPOCHS / epoch_divide_by) < GLOBAL_EPOCHS:
            # Train <GLOBAL_EPOCHS / epoch_divide_by> epoches

            image, _ = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
            result = model.predict(image)
            postprocess_non_nms_result(image, result, no_suppress=False, display_all=True)

            model.fit(
                TARGET_TRAIN_DATA,
                epochs=int(GLOBAL_EPOCHS / epoch_divide_by),
                validation_data=TARGET_TRAIN_DATA,
                shuffle=True,
                callbacks=[save_best_model],
                verbose=1
            )

            epoch_iteration += 1
    else:
        model.fit(
            TARGET_TRAIN_DATA,
            epochs=GLOBAL_EPOCHS,
            validation_data=TARGET_TRAIN_DATA,
            shuffle=True,
            callbacks=[save_best_model] if MODEL_SAVE else None,
            verbose=1
        )
else:
    import random

    data_iterations = 4
    for _ in range(data_iterations):
        image, label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
        result = model.predict(image)
        # postprocess_calculate_precision(result, label)
        postprocess_non_nms_result(image, result, no_suppress=False)



