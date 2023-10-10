#!/usr/bin/env python3
"""contains the yolo class"""
import tensorflow as tf
import numpy as np


class Yolo():
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r") as myfile:
            self.class_names = myfile.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors


    def process_outputs(self, outputs, image_size):
        """
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        boxes -> contains the processed boundary boxes
        box_confidences -> contains the box confidences for each output
        box_class_probs -> contains the box class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        ih = image_size[0]
        iw = image_size[1]
        print(ih, iw)
        for output in outputs:
            output[..., :2] = 1.0 / (1.0 + np.exp(-output[..., :2]))
            output[..., 4] = 1.0 / (1.0 + np.exp(-output[..., 4]))
            gh = output.shape[0]
            gw = output.shape[1]
            split_output = np.array_split(output, (4, 5, ), axis=3)




            test = split_output[0][0][0][0]
            answer = [-2.13743365e+02, -4.85478868e+02,  3.05682061e+02, 5.31534670e+02]
            print(test)
            print(answer[2]- answer[0])
            print((answer[2] - answer[0]))
            print((answer[3] - answer[1]))





            box_confidences.append(split_output[1])
            box_class_probs.append(split_output[2])
        return ((boxes, box_confidences, box_class_probs))
