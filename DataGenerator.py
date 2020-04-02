from tensorflow.keras import utils
import math
import numpy as np
import cv2
from Decoder import LabelDecoder

import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image


def test_augmented_items(image_aug, bbs_aug):
    bbs_aug = bbs_aug.remove_out_of_image()
    bbs_aug = bbs_aug.clip_out_of_image()

    Image.fromarray(bbs_aug.draw_on_image(np.array(image_aug)), 'RGB').show()
    pass


class Labeler():
    def __init__(self, names_filename):
        self.names_list = {}

        with open(names_filename) as f:
            idx = 0
            for line in f:
                self.names_list[idx] = line
                idx += 1

    def get_name(self, index):
        return self.names_list[index]


class Dataloader(utils.Sequence):
    DEFAULT_AUGMENTER = iaa.SomeOf(2, [
    iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
    iaa.Affine(
            translate_px={"x": 3, "y": 10},
            scale=(0.9, 0.9)
    ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    iaa.AdditiveGaussianNoise(scale=0.1 * 255),
    iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
    iaa.Affine(rotate=45),
    iaa.Sharpen(alpha=0.5)
])

    def __init__(self, file_name, dim=(448, 448, 3), batch_size=1, numClass=1, augmentation=False, shuffle=True):
        self.image_list, self.label_list = self.GetDataList(file_name)
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter = self.DEFAULT_AUGMENTER if augmentation else False
        self.augmenter_size = 4
        self.outSize = 5 + numClass
        self.labeler = Labeler('voc.names')
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_list) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.image_list[k] for k in indexes]
        batch_y = [self.label_list[k] for k in indexes]

        X, y = self.__data_generation(batch_x, batch_y)

        # 마지막에 [None]을 넣는 것은... 다른 버전에서 동작하지 않는다
        return np.asarray(X), np.asarray(y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def GetDataList(self, folder_path: str):
        train_list = []
        lable_list = []

        f = open(folder_path, 'r')
        while True:
            line = f.readline()
            if not line: break
            train_list.append(line.replace("\n", ""))
            label_text = line.replace(".jpg", ".txt")
            label_text = label_text.replace(
                'C:\\Users\\jungin500\\Desktop\\Study\\2020-yolov3-impl\\VOCdevkit\\VOC2007\\JPEGImages\\',
                "C:\\Users\\jungin500\\Desktop\\Study\\2020-yolov3-impl\\VOCyolo\\")
            label_text = label_text.replace("\n", "")
            lable_list.append(label_text)

        return train_list, lable_list

    def __convert_yololabel_to_iaabbs(self, yolo_raw_label, image_width=448, image_height=448):
        # raw_label = [bboxes, 5], np.array([center_x, center_y, w, h, c])
        return ia.BoundingBoxesOnImage([
            ia.BoundingBox(
                x1=yolo_raw_bbox[0],
                y1=yolo_raw_bbox[1],
                x2=yolo_raw_bbox[2],
                y2=yolo_raw_bbox[3],
                # label=class_list[int(yolo_bbox[0])] # Label을 id로 활용하자
                label=yolo_raw_bbox[4]
            ) for yolo_raw_bbox in yolo_raw_label
        ], shape=(image_width, image_height))

    def __convert_iaabbs_to_yololabel(self, iaa_bbs_out):
        label = np.zeros((7, 7, 25), dtype=np.float32)
        raw_label = []

        for bbox in iaa_bbs_out.bounding_boxes:
            center_x = bbox.center_x / 448
            center_y = bbox.center_y / 448
            width = bbox.width / 448
            height = bbox.height / 448
            class_id = int(float(bbox.label))  # Explicit

            scale_factor = (1 / 7)

            grid_x_index = int(center_x // scale_factor)
            grid_y_index = int(center_y // scale_factor)
            grid_x_index, grid_y_index = \
                np.clip([grid_x_index, grid_y_index], a_min=0, a_max=6)

            relative_coord = [center_x * 7, center_y * 7]
            relative_center_x = relative_coord[0] - int(relative_coord[0])
            relative_center_y = relative_coord[1] - int(relative_coord[1])

            label[grid_y_index][grid_x_index][class_id] = 1.
            label[grid_y_index][grid_x_index][20:] = np.array([relative_center_x, relative_center_y, width, height, 1])

            raw_label.append(np.array([center_x, center_y, width, height, class_id]))

        return label, np.array(raw_label)

    def __data_generation(self, list_img_path, list_label_path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.augmenter:
            X = np.empty((self.batch_size * self.augmenter_size, *self.dim))
            Y = np.empty((self.batch_size * self.augmenter_size, *(7, 7, 25)))
        else:
            X = np.empty((self.batch_size, *self.dim))
            Y = np.empty((self.batch_size, *(7, 7, 25)))

        # Generate data
        for i, path in enumerate(list_img_path):
            original_image = (np.array(Image.open(path).resize((448, 448))) / 255).astype(np.float32)

            # raw_label은 x_1, y_1, x_2, y_2, c를 가지고 있다.
            label, raw_label = self.GetLabel(list_label_path[i], original_image.shape[0], original_image.shape[1])
            if self.augmenter:
                iaa_bbs = self.__convert_yololabel_to_iaabbs(raw_label)
                for aug_idx in range(self.augmenter_size - 1):
                    augmented_image, augmented_label = self.augmenter(
                        image=(original_image * 255).astype(np.uint8),
                        bounding_boxes=iaa_bbs
                    )
                    # test_augmented_items(augmented_image, augmented_label)
                    X[aug_idx * self.batch_size + i,] = augmented_image / 255
                    Y[aug_idx * self.batch_size + i,], augmented_raw_label = \
                        self.__convert_iaabbs_to_yololabel(augmented_label.remove_out_of_image().clip_out_of_image())

                # 마지막 items는 non-augmented images이다.
                X[self.augmenter_size - 1 + i,] = original_image
                Y[self.augmenter_size - 1 + i,] = label
            else:
                X[i,] = original_image
                Y[i,] = label

        return X, Y

    def GetLabel(self, label_path, img_h, img_w):
        f = open(label_path, 'r')
        label = np.zeros((7, 7, 25), dtype=np.float32)
        raw_label = []
        while True:
            line = f.readline()
            if not line: break

            dw = 1. / img_w
            dh = 1. / img_h

            split_line = line.split()

            c, x, y, w, h = split_line

            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            c = int(c)

            scale_factor = (1 / 7)

            # // : 몫
            grid_x_index = int(x // scale_factor)
            grid_y_index = int(y // scale_factor)

            # x, y는 해당 셀 내에서의
            # 상대적 위치를 가지고 간다.
            grid_idx = [x * 7, y * 7]
            x_relative = grid_idx[0] - int(grid_idx[0])
            y_relative = grid_idx[1] - int(grid_idx[1])

            label[grid_y_index][grid_x_index][c] = 1.
            label[grid_y_index][grid_x_index][20:] = np.array([x_relative, y_relative, w, h, 1])

            raw_label.append(np.array([
                int((x - w / 2) * 448),
                int((y - h / 2) * 448),
                int((x + w / 2) * 448),
                int((y + h / 2) * 448),
                c
            ]))

        return label, np.array(raw_label)

    def GetLabelName(self, label_id):
        return self.labeler.get_name(label_id)
