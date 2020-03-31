from abc import ABC

import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence


class VOC2007Generator(Sequence, ABC):
    IMAGE_FOLDER = 'JPEGImages'
    LABEL_FOLDER = 'Annotations'
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, train, root_path, batch_size=8, shuffle=True, class_path='./voc.names'):
        self.root_path = root_path
        self.train = train  # "train" 또는 "test"
        self.batch_size = batch_size  # 배치로 처리하기 위함
        self.shuffle = shuffle
        self.class_path = class_path

        with open(class_path) as f:
            self.classes = f.read().splitlines()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

    def on_epoch_end(self):
        pass

    def _check_exists(self):
        print("Image folder: {}".format(os.path.join(self.root_path, self.IMAGE_FOLDER)))
        print("Label folder: {}".format(os.path.join(self.root_path, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root_path, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root_path, self.LABEL_FOLDER))

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
