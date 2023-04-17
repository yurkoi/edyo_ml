from scipy.spatial.distance import cdist

from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
import tensorflow as tf


def poses_diff(x):
    H, W = x.get_shape()[1], x.get_shape()[2]
    x = tf.subtract(x[:, 1:, ...], x[:, :-1, ...])
    x = tf.image.resize(x, size=[H, W])
    return x


def pose_motion(P, frame_l):
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l, -1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:, ::2, ...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l / 2), -1))(P_diff_fast)
    return P_diff_slow, P_diff_fast


def c1D(x, filters, kernel):
    x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def block(x, filters):
    x = c1D(x, filters, 3)
    x = c1D(x, filters, 3)
    return x


def d1D(x, filters):
    x = Dense(filters, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def build_FM(frame_l=32, joint_n=22, joint_d=2, feat_d=231, filters=16):
    M = Input(shape=(frame_l, feat_d))
    P = Input(shape=(frame_l, joint_n, joint_d))

    diff_slow, diff_fast = pose_motion(P, frame_l)

    x = c1D(M, filters * 2, 1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 3)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x_d_slow = c1D(diff_slow, filters * 2, 1)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 3)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)

    x_d_fast = c1D(diff_fast, filters * 2, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 3)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)

    x = concatenate([x, x_d_slow, x_d_fast])
    x = block(x, filters * 2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 8)
    x = SpatialDropout1D(0.1)(x)

    return Model(inputs=[M, P], outputs=x)


def build_DD_Net(frame_l=32, joint_n=22, joint_d=3, feat_d=231, clc_num=14, filters=16):
    M = Input(name='M', shape=(frame_l, feat_d))
    P = Input(name='P', shape=(frame_l, joint_n, joint_d))

    FM = build_FM(frame_l, joint_n, joint_d, feat_d, filters)

    x = FM([M, P])

    x = GlobalMaxPool1D()(x)

    x = d1D(x, 128)
    x = Dropout(0.5)(x)
    x = d1D(x, 128)
    x = Dropout(0.5)(x)
    x = Dense(clc_num, activation='softmax')(x)

    ######################Self-supervised part
    model = Model(inputs=[M, P], outputs=x)
    return model


def convert_mediapipe_for_DDNet(hand):
    c1_x = (hand[0, 0] + hand[9, 0]) / 2
    c1_y = (hand[0, 1] + hand[9, 1]) / 2
    c1_z = (hand[0, 2] + hand[9, 2]) / 2

    c2_x = (hand[0, 0] + hand[13, 0]) / 2
    c2_y = (hand[0, 1] + hand[13, 1]) / 2
    c2_z = (hand[0, 2] + hand[13, 2]) / 2

    c_x = (c1_x + c2_x) / 2
    c_y = (c1_y + c2_y) / 2
    c_z = (c1_z + c2_z) / 2
    new_hand = np.insert(hand, 1, [c_x, c_y, c_z], axis=0)

    dist = cdist(new_hand, new_hand)
    JCD = []
    for i, d in enumerate(dist):
        JCD.extend(dist.T[i, i + 1:])

    return JCD, new_hand


class Config():
    def __init__(self):
        self.frame_l = 32 # the length of frames
        self.joint_n = 12 # the number of joints
        self.joint_n = 22 # the number of joints
        self.joint_d = 3 # the dimension of joints
        self.clc_coarse = 14 # the number of coarse class
        self.clc_fine = 28 # the number of fine-grained class
        self.feat_d = 231
        self.filters = 64
C = Config()

LABELS = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation Clockwise', 'Rotation Counter Clockwise',
          'Swipe Right', 'Swipe Left', 'Swipe Up', 'Swipe Down', 'Swipe X', 'Swipe +', 'Swipe V', 'Shake']