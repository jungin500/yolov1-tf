from tensorflow.keras import utils
import math
import numpy as np
import cv2
from Decoder import LabelDecoder


class Dataloader(utils.Sequence):
    def __init__(self, file_name, dim=(448, 448, 3), batch_size=1, numClass=1, shuffle=True):
        self.image_list, self.lable_list = self.GetDataList(file_name)
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.outSize = 5 + numClass
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_list) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.image_list[k] for k in indexes]
        batch_y = [self.lable_list[k] for k in indexes]

        X, y = self.__data_generation(batch_x, batch_y)

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
            train_list.append(line)
            label_text = line.replace(".jpg", ".txt")
            lable_list.append(label_text)

        return train_list, lable_list

    def __data_generation(self, list_img_path, list_label_path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *(7, 7, 6)))

        # Generate data
        for i, path in enumerate(list_img_path):
            # Store sample
            _img = cv2.imread(path)
            img = cv2.resize(_img, (448, 448))
            # cv2.imshow('Image', img)
            # cv2.waitKey(0)

            X[i,] = img / 255.

            label = self.GetLabel(list_label_path[i], _img.shape[0], _img.shape[1])
            y[i,] = label

        return X, y

    def GetLabel(self, label_path, img_h, img_w):
        f = open(label_path, 'r')
        label = np.zeros((7, 7, 6), dtype=np.float32)
        while True:
            line = f.readline()
            if not line: break

            dw = 1. / img_w
            dh = 1. / img_h

            split_line = line.split()

            x, y, w, h, c = split_line

            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            c = float(c)

            center_x = x + (w / 2.0)
            center_y = y + (h / 2.0)

            center_x = center_x * dw
            center_y = center_y * dh

            w = w * dw
            h = h * dh

            scale_factor = (1 / 7)
            # // : 몫
            grid_x_index = int(center_x // scale_factor)
            grid_y_index = int(center_y // scale_factor)

            x_offset = (center_x / scale_factor) - grid_x_index
            y_offset = (center_y / scale_factor) - grid_y_index

            label[grid_y_index][grid_x_index] = np.array([1, x_offset, y_offset, w, h, c])

        return label
