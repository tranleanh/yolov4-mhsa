from functools import wraps

import tensorflow as tf

from keras import backend as K
from keras.initializers import random_normal, constant
from keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Layer, ZeroPadding2D, LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import Reshape, Dot, Permute, Softmax
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from utils.utils import compose


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


#------------------------------------------------------#
#   DarknetConv2D
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


#---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + Mish
#---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


#---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def convolutional(input_layer, filters_shape, downsample=False, activate=True, activate_type='leaky'):
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    if activate:
        if activate_type == "mish":
            conv = DarknetConv2D_BN_Mish(filters_shape[-1], (filters_shape[0],filters_shape[0]), strides=(strides,strides))(input_layer)

        elif activate_type == "leaky":
            conv = DarknetConv2D_BN_Leaky(filters_shape[-1], (filters_shape[0],filters_shape[0]), strides=(strides,strides))(input_layer)
    else:
        conv = DarknetConv2D(filters_shape[-1], (filters_shape[0],filters_shape[0]), strides=(strides,strides))(input_layer)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    # residual_output = short_cut + conv
    residual_output = Add()([short_cut, conv])

    return residual_output


#---------------------------------------------------#
#   CSPDarknet53 Backbone
#---------------------------------------------------#
def cspdarknet53_backbone(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = Concatenate()([input_data, route])
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    input_data = Concatenate()([MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(input_data), \
                                MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(input_data),  \
                                MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(input_data), \
                                input_data])

    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data


#---------------------------------------------------#
#   MHSA Block
#---------------------------------------------------#
def mhsa_block(input_layer, input_channel):

    # W, H = 25, 25
    W, H = int(input_layer.shape[1]), int(input_layer.shape[2])

    # From 2-D to Sequence: WxHxd -> W*Hxd (e.g., 25x25x512 -> 1x625x512)
    conv = Reshape((1, W*H, input_channel))(input_layer)

    # Position Encoding: 1x625x512 -> 1x625x512
    pos_encoding = Conv2D(input_channel, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    # Element-wise Sum: 1x625x512
    conv = Add()([conv, pos_encoding])

    # Query: Conv1x1 --> 1x625x512
    conv_q = Conv2D(input_channel, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    # Key: Conv1x1 --> 1x625x512
    conv_k = Conv2D(input_channel, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    # Value: Conv1x1 --> 1x625x512
    conv_v = Conv2D(input_channel, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    # Transposed Key: 1x512x612
    conv_k = Permute(dims=(1, 3, 2))(conv_k)

    # Content-content: Query * Key_T --> 1x625x625
    conv = Dot(axes=(3,2))([conv_q, conv_k])
    conv = Reshape((1, W*H, W*H))(conv)

    # Softmax --> 1x625x625
    conv = Softmax()(conv)

    # Output: Dot(1x625x625, 1x625x512) --> 1x625x512
    conv = Dot(axes=(3,2))([conv, conv_v])

    # From Sequence to 2-D
    conv = Reshape((W, H, input_channel))(conv)

    return conv


#---------------------------------------------------#
#   CSPDarknet-MHSA Backbone
#---------------------------------------------------#
def mhsa_cspdarknet53_backbone(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = Concatenate()([input_data, route])
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")

    # MHSA Blocks
    for i in range(4):
        input_data = mhsa_block(input_data, 512)

    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = Concatenate()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    input_data = Concatenate()([MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(input_data), \
                                MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(input_data),  \
                                MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(input_data), \
                                input_data])

    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data