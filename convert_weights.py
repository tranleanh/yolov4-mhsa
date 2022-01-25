import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

# from nets.yolo import get_train_model, yolo_body

# from nets.yolo import get_train_model
from nets.yolo import yolo_body_mhsa as yolo_body

from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             WarmUpCosineDecayScheduler)

from utils.dataloader import YoloDatasets
from utils.utils import get_anchors, get_classes
from utils.utils import load_weights

if __name__ == "__main__":

    #--------------------------------------------------------#
    #   Be sure to modify classes_path before training so that it corresponds to your own dataset.
    #--------------------------------------------------------#
    classes_path    = 'model_data/waymo_classes.txt'

    #---------------------------------------------------------------------#
    #   anchors_path: represents the txt file corresponding to the anchors, which is generally not modified.
    #   anchors_mask: is used to help the code find the corresponding anchors and is generally not modified.
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/waymo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    #----------------------------------------------------------------------------------------------------------------------------#
    #   
    #   If you want the model to start training from scratch, set model_path = '', Freeze_Train = Fasle, 
    #   Then start training from scratch, and there is no process of freezing the backbone.
    #   
    #   Generally, performance of training from scratch will be poor, because the weights are too random.
    #
    #   The network generally should not be trained from scratch, at least the weights of the backbone part are used. 
    #   Some papers mention that training from scratch is not necessary.
    #   
    #   -> Training from scratch is not recommended.
    #
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = './model_data/yolov4-waymo100k_800_best.weights'

    #------------------------------------------------------#
    #   The size of the input shape must be a multiple of 32
    #------------------------------------------------------#
    # input_shape     = [416, 416]

    #------------------------------------------------------#
    #   Training settings:
    #   mosaic data augmentation: True or False (mosaic data enhancement is not stable, so default setting is False)
    #   Cosine_scheduler: True or False
    #   label_smoothing: generally below 0.01
    #------------------------------------------------------#
    mosaic              = False
    Cosine_scheduler    = False
    label_smoothing     = 0.01

    #----------------------------------------------------#
    #   The training is divided into two phases: the freezing phase and the non-freezing phase.
    #   When the memory error occurs, reduce the batch size
    #   Due to BatchNorm layer, the minimum batch_size is 2 and cannot be 1.
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   During freezing phase parameters, the backbone weights will not be changed
    #   Occupy less memory, only fine-tune the network
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 2
    Freeze_batch_size   = 4
    Freeze_lr           = 1e-3
    # Freeze_lr           = 1e-5
    #----------------------------------------------------#
    #   Fine-tune whole network
    #   The weights of whole network are updated
    #   Occupy more memory
    #----------------------------------------------------#
    UnFreeze_Epoch      = 40
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-4
    # Unfreeze_lr         = 1e-5
    #------------------------------------------------------#
    #   Whether to apply freezing or non-freezing
    #------------------------------------------------------#
    Freeze_Train        = False
    #------------------------------------------------------#
    #   Whether to use multi workers
    #   When enable it, it will speed up data reading, but it will take up more memory
    #   When multi workers training is enabled in keras, sometimes the speed is much slower
    #------------------------------------------------------#
    num_workers         = 1

    #----------------------------------------------------#
    #   Training data paths
    #----------------------------------------------------#
    train_annotation_path   = '2007_train_old.txt'
    val_annotation_path     = '2007_val_old.txt'

    #----------------------------------------------------#
    #   Load classes and anchors
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    K.clear_session()

    #------------------------------------------------------#
    #   Create model and load weights
    #------------------------------------------------------#
    img_size = 800
    model_body  = yolo_body((img_size, img_size, 3), anchors_mask, num_classes)
    model_body.summary()

    # load_weights(model_body, model_path)

    # print("Loading Weights Done!")

    # model_body.save("yolov4_waymo100k_converted.h5")

    # if model_path != '':
    #     print('Load weights {}.'.format(model_path))
    #     model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
