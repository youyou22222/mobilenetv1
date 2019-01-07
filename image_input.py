""" Build an Image Dataset in TensorFlow.

For this example, you need to make your own set of images (JPEG).
We will show 2 different ways to build that dataset:

- From a root folder, that will have a sub-folder containing images for each class
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```

- From a plain text file, that will list all images with their class ID:
    ```
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    ```
"""

from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

import os
import cv2
import random
import numpy as np
import scipy
DATA_PATH="./data/"
IMAGE_HEIGHT=64
IMAGE_WIDTH=64


class Dataset(object):
    def __init__(self, datapath=DATA_PATH):
        self.data_path = datapath
        self.image_path = []
        self.label = []
        self.num_classes = 0
        for classes in os.walk(self.data_path).__next__()[1]:

            subdir = os.path.join(self.data_path, classes)
            self.num_classes = self.num_classes+1
            for img in os.walk(subdir).__next__()[2]:
                imgpath = os.path.join(subdir, img)
                self.image_path.append(imgpath)
                self.label.append(int(classes)-1)
        assert len(self.image_path) == len(self.label)


    def generate_batch(self, batch_size=64, shuffle=True):

        imagepaths = tf.convert_to_tensor(self.image_path, dtype=tf.string)
        labels = tf.convert_to_tensor(self.label, dtype=tf.int64)
        image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                     shuffle=shuffle)

        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.cast(image, tf.float32)
        #image = tf.image.per_image_standardization(image)
        #image = (255-image)/255
        image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        if shuffle:
            image, label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch_size,
                                                   capacity=batch_size*8,
                                                  min_after_dequeue=batch_size,
                                                  num_threads=4)
            label = slim.one_hot_encoding(label, self.num_classes - 0)

        else:
            image, label = tf.train.batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity=batch_size*8)
        return image, label

def hcl_input(is_training=True, datapath="/data/home/jyw/hcl_cassia_new/train_test"):
    print("train dataset dir: {}".format(datapath))
    data = Dataset(datapath=datapath)
    return data.generate_batch(shuffle=is_training)

def DataSet():
    with tf.Session() as sess:
        dataset = Dataset("/data/home/jyw/hcl_cassia/for_test")
        images, labels = dataset.generate_batch(shuffle=False)
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            # in most cases coord.should_stop() will return True
            # when there are no more samples to read
            # if num_epochs=0 then it will run for ever
            while not coord.should_stop():
                # will start reading, working data from input queue
                # and "fetch" the results of the computation graph
                # into raw_images and raw_labels
                raw_images, raw_labels = sess.run([images, labels])
                for img in raw_images:
                    print(img.dtype)
                    cv2.imshow("test", img)
                    cv2.waitKey(0)
                for label in raw_labels:
                    print(label.shape)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    DataSet()
