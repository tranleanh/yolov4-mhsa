import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import glob
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [(x.strip()).split() for x in content]
    return content


def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


if __name__ == "__main__":

    datapath = "waymo_val.txt"
    data = file_lines_to_list(datapath)

    dir_save_path   = "texts/yolov4_keras_waymo_ep13_resume_2"

    if len(data) > 0:

        if not os.path.isdir(dir_save_path):
            os.makedirs(dir_save_path)

        print(f"Total test images: {len(data)}")
        yolo = YOLO()

        for i, line in enumerate(data):

            path = line[0]
            fname = get_file_name(path)
            image = Image.open(path)
            preds = yolo.detect_and_write_txt(image)

            print(i, len(data), fname)

            txt_file = open(f"{dir_save_path}/{fname}.txt", "w")

            for pred in preds: 
                cls_name = pred[0]
                score = pred[1]
                top, left, bottom, right = pred[2], pred[3], pred[4], pred[5] 
                print(cls_name, score, left, top, right, bottom, file=txt_file)

            txt_file.close()

            # if i == 5: break

    else:
        print("There is no data!")

    print("DONE!")

 