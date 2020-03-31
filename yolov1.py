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
