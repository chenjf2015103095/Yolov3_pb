# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :       陈剑锋
   Date：         2019-08-28 上午9:50
   Description :  Dream it possible!
   
-------------------------------------------------
   Change Activity:

-------------------------------------------------
"""
import cv2
import os
import numpy as np
import core.utils as utils
import tensorflow as tf

class PredictNode(object):

    def __init__(self,model_path,num_classes,input_size,return_elements):
        self.model_path=model_path
        self.num_classes=num_classes
        self.input_size=input_size
        self.return_elements=return_elements
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.conf = tf.ConfigProto()
        self.conf.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.conf.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.return_tensors = utils.read_pb_return_tensors(self.graph, self.model_path, self.return_elements)
        self.sess = tf.Session(graph=self.graph, config=self.conf)

    def predict(self,image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(np.copy(original_image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        with tf.Session(graph=self.graph) as sess:
            pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
                [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
                        feed_dict={ self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, self.input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image,label = utils.draw_bbox(original_image, bboxes)
        # image = Image.fromarray(image)
        # print(image)
        cv2.imwrite('static/images/test.jpg',image)
        # image.show()
        return label



if __name__=='__main__':
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    model_path = "/media/ubutnu/fc1a3be7-9b03-427e-9cc9-c4b242cefbff/Yolo/tensorflow-yolov3/yolov3_coco.pb"
    num_classes = 20
    input_size = 416
    Eval = PredictNode(model_path, num_classes, input_size, return_elements)
    image_path='/media/ubutnu/fc1a3be7-9b03-427e-9cc9-c4b242cefbff/Yolo/tensorflow-yolov3/docs/images/001.jpg'
    label=Eval.predict(image_path)