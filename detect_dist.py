import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

from yolov3_tf2.dist_model import *

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        distNN = DistNNTiny()
    else:
        distNN = YoloV3(classes=FLAGS.num_classes)

    distNN.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    circles, objectness = distNN(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    maxObj = max(objectness[0])
    for i in range(tf.shape(objectness)[1]):
        if objectness[0][i] == maxObj:
            logging.info('\tDrone, {}, {}'.format(np.array(objectness[0][i]),
                                                  np.array(circles[0][i])))
            circle = np.array(circles[0][i])
            obj = np.array(objectness[0][i])

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_circle(img, circle, obj)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))


def draw_circle(img, circle, obj):
    lado = circle[2]
    cx = circle[0] + lado / 2
    cy = circle[1] + lado / 2
    # ima = cv2.circle(img, (
    #     int(circle[0] * np.shape(img)[1]), int(circle[1] * np.shape(img)[0])),
    #                  int(circle[2] * np.shape(img)[1]), (255, 0, 0), 2)
    ima = cv2.circle(img,
                     (int(cx * np.shape(img)[1]), int(cy * np.shape(img)[0])),
                     int(lado / 2 * np.shape(img)[1]), (255, 0, 0), 2)

    return ima


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
