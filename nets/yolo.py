from keras.layers import BatchNormalization, Concatenate, Conv2D, Input, \
                            Lambda, LeakyReLU, MaxPooling2D, UpSampling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization
from keras.models import Model
from utils.utils import compose

from nets.CSPdarknet53 import convolutional
from nets.CSPdarknet53 import cspdarknet53_backbone
from nets.CSPdarknet53 import mhsa_cspdarknet53_backbone
from nets.yolo_training import yolo_loss


def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask, 
                           'num_classes' : num_classes, 'label_smoothing' : label_smoothing}
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model


#---------------------------------------------------#
#   YOLOv4
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes):
    inputs      = Input(input_shape)
    
    route_1, route_2, conv = cspdarknet53_backbone(inputs)

    route = conv
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = UpSampling2D(2)(conv)
    route_2 = convolutional(route_2, (1, 1, 512, 256))
    conv = Concatenate()([route_2, conv])

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = UpSampling2D(2)(conv)
    route_1 = convolutional(route_1, (1, 1, 256, 128))
    conv = Concatenate()([route_1, conv])

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (num_classes + 5)), activate=False)

    conv = convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = Concatenate()([conv, route_2])

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (num_classes + 5)), activate=False)

    conv = convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = Concatenate()([conv, route])

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (num_classes + 5)), activate=False)

    return Model(inputs, [conv_lbbox, conv_mbbox, conv_sbbox])


#---------------------------------------------------#
#   YOLOv4-MHSA
#---------------------------------------------------#
def yolo_mhsa(input_shape, anchors_mask, num_classes):
    inputs      = Input(input_shape)
    
    route_1, route_2, conv = mhsa_cspdarknet53_backbone(inputs)

    route = conv
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = UpSampling2D(2)(conv)
    route_2 = convolutional(route_2, (1, 1, 512, 256))
    conv = Concatenate()([route_2, conv])

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = UpSampling2D(2)(conv)
    route_1 = convolutional(route_1, (1, 1, 256, 128))
    conv = Concatenate()([route_1, conv])

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (num_classes + 5)), activate=False)

    conv = convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = Concatenate()([conv, route_2])

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (num_classes + 5)), activate=False)

    conv = convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = Concatenate()([conv, route])

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (num_classes + 5)), activate=False)

    return Model(inputs, [conv_lbbox, conv_mbbox, conv_sbbox])




