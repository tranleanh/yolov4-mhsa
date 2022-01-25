import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from nets.yolo import get_train_model

from nets.yolo import yolo_mhsa as yolo_body

from utils.callbacks import ExponentDecayScheduler, LossHistory, WarmUpCosineDecayScheduler

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
    # model_path      = 'model_data/yolov4-waymo100k_800_best.weights'
    model_path      = 'model_data/yolov4_waymo100k_ep040-loss2.874-val_loss3.338.h5'

    #------------------------------------------------------#
    #   The size of the input shape must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [800, 800]

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
    Freeze_Epoch        = 1
    Freeze_batch_size   = 2
    Freeze_lr           = 1e-4
    # Freeze_lr           = 1e-5
    #----------------------------------------------------#
    #   Fine-tune whole network
    #   The weights of whole network are updated
    #   Occupy more memory
    #----------------------------------------------------#
    UnFreeze_Epoch      = 50
    Unfreeze_batch_size = 2
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
    train_annotation_path   = 'waymo_train.txt'
    val_annotation_path     = 'waymo_val_mini100.txt'


    #----------------------------------------------------#
    #   Load classes and anchors
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    K.clear_session()
    #------------------------------------------------------#
    #   Create model and load weights
    #------------------------------------------------------#
    input_size = input_shape[0]
    model_body  = yolo_body((input_size, input_size, 3), anchors_mask, num_classes)
    model_body.summary()

    #----------------------------------------------------#
    # Load Weights

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    # if model_path != '':
    #     load_weights(model_body, model_path)
    #     print("Loading Weights Done!")
    #----------------------------------------------------#

    model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing)

    #-------------------------------------------------------------------------------#
    #   Training parameters
    #   logging indicates the storage address of tensorboard
    #   checkpoint is used to set the details of weight saving, period is used to modify how many epochs are saved once
    #   reduce_lr: learning rate decay
    #   early_stopping: training will automatically stop when the val_loss does not drop, indicating that the model has basically converged
    #-------------------------------------------------------------------------------#
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    if Cosine_scheduler:
        reduce_lr   = WarmUpCosineDecayScheduler(T_max = 5, eta_min = 1e-5, verbose = 1)
    else:
        reduce_lr   = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)

    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    loss_history    = LossHistory('logs/')

    #---------------------------#
    #   Read dataset
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if Freeze_Train:
        freeze_layers = 200
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
        
    #------------------------------------------------------#
    #   Freezing backbone
    #   To prevent weights from being corrupted in the early stages of training.
    #   Init_Epoch: starting epoch
    #   Freeze_Epoch: end of freezing
    #   UnFreeze_Epoch: total num of training epochs
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')
        
        model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, mosaic = False, train = False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )

    if Freeze_Train:
        for i in range(freeze_layers): model_body.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')
        
        model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, mosaic = False, train = False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
