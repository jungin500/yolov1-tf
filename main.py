# INFO까지의 로그 Suppress하기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.optimizers import Adam
from yolov1 import Yolov1Model, Yolov1Loss, INPUT_LAYER
from DataGenerator import Dataloader

train_data = Dataloader(file_name='train_list.txt', numClass=20)

model = Yolov1Model()
optimizer = Adam(learning_rate=1e-5, decay=5e-4)

model.compile(optimizer=optimizer, loss=Yolov1Loss)
# model.summary()

model.fit(train_data, epochs=100, validation_data=None, shuffle=False)