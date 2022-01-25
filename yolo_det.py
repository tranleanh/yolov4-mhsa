import colorsys
import os
import time

import numpy as np
from keras import backend as K
from PIL import ImageDraw, ImageFont

# from nets.yolo import yolo_body
from nets.yolo import yolo_body_mhsa as yolo_body

from utils.utils import cvtColor, get_anchors, get_classes, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {
        #------------------------------------------------- -------------------------#
        # To use your own trained model for prediction, you must modify model_path and classes_path!
        # model_path points to the weights file under the logs folder, classes_path points to the txt under model_data
        #
        # After training, there are multiple weight files in the logs folder, and you can select the validation set with lower loss.
        # The lower loss of the validation set does not mean that the mAP is higher, it only means that the weight has better generalization performance on the validation set.
        # If the shape does not match, pay attention to the modification of the model_path and classes_path parameters during training
        #------------------------------------------------- -------------------------#
        "model_path"        : 'logs/ep040-loss36.271-val_loss35.148.h5',
        "classes_path"      : 'model_data/chess_classes.txt',
        #---------------------------------------------------------------------#
        # anchors_path represents the txt file corresponding to the anchors, which is generally not modified.
        # anchors_mask is used to help the code find the corresponding anchors and is generally not modified.
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        # Enter the size of the image, which must be a multiple of 32.
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        # ============> Change input shape below <===============
        #---------------------------------------------------------------------#
        # Only prediction boxes with scores greater than confidence will be kept
        #---------------------------------------------------------------------#
        "confidence"        : 0.05,
        #---------------------------------------------------------------------#
        # nms_iou: used for non-maximum suppression
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.45,
        #---------------------------------------------------------------------#
        # maximum number of boxes
        #---------------------------------------------------------------------#
        "max_boxes"         : 100,
        #---------------------------------------------------------------------#
        # This variable is used to control whether to use letterbox_image to resize the input image without distortion,
        # After many tests, it is found that the direct resize effect of closing letterbox_image is better
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   initialize yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        # Get the numbers of categories and anchors
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        #---------------------------------------------------#
        # Set color
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.input_image_shape = K.placeholder(shape=(2, ))

        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    #---------------------------------------------------#
    #   load model
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        input_size = 416
        self.yolo_model = yolo_body([input_size, input_size, 3], self.anchors_mask, self.num_classes)
        self.yolo_model.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))
        #---------------------------------------------------------#
        # In the yolo_eval function, we will post-process the prediction results
        # The content of post-processing includes decoding, non-maximum suppression, threshold filtering, etc.
        #---------------------------------------------------------#
        boxes, scores, classes = DecodeBox(
            self.yolo_model.output, 
            self.anchors,
            self.num_classes, 
            self.input_image_shape, 
            self.input_shape, 
            anchor_mask     = self.anchors_mask,
            max_boxes       = self.max_boxes,
            confidence      = self.confidence, 
            nms_iou         = self.nms_iou, 
            letterbox_image = self.letterbox_image
        )
        return boxes, scores, classes

    #---------------------------------------------------#
    #   Detect in pictures
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        # The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        # Add gray padding to the image to achieve undistorted resize
        # You can also directly resize for identification
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        # Add the batch_size and normalize it
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   Feed the image into the network to make predictions
        #---------------------------------------------------------#
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #---------------------------------------------------------#
        #   Set font and box thickness
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #---------------------------------------------------------#
        #   image drawing
        #---------------------------------------------------------#
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


    def detect_and_write_txt(self, image):

        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        # font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        preds = []

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            preds.append([predicted_class, score, top, left, bottom, right])

            # label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)
            # label = label.encode('utf-8')

            # print(predicted_class, score, top, left, bottom, right)


            
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            # for i in range(thickness):
            #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            # del draw

        return preds


    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        # The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        # Add gray bars to the image to achieve undistorted resize
        # You can also directly resize for identification
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        # Add the batch_size dimension and normalize it
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        #---------------------------------------------------------#
        # Feed the image into the network for prediction!
        #---------------------------------------------------------#
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0})
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        # Add gray bars to the image to achieve undistorted resize
        # You can also directly resize for identification
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        # Add the batch_size dimension and normalize it
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        for i, c in enumerate(out_classes):
            predicted_class             = self.class_names[int(c)]
            score                       = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

    def close_session(self):
        self.sess.close()
