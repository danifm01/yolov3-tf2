from yolov3_tf2.models import *

from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

distNN_tiny_anchors = np.array([12, 25, 42, 81, 152, 322], np.float32) / 416
distNN_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def dist_circles(pred, anchors):
    # pred: (batch_size, grid, grid, anchors, (x, y, r))
    grid_size = tf.shape(pred)[1]
    center_xy, radius, objectness = tf.split(
        pred, (2, 1, 1), axis=-1)

    center_xy = tf.sigmoid(center_xy)
    objectness = tf.sigmoid(objectness)
    pred_circle = tf.concat((center_xy, radius), axis=-1)

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    center_xy = (center_xy + tf.cast(grid, tf.float32)) / \
                tf.cast(grid_size, tf.float32)

    # REVISAR: POSIBLE FUENTE DE ERRROR
    radius = tf.exp(radius) * np.reshape(anchors, (3, 1))

    circles = tf.concat([center_xy, radius], axis=-1)

    return circles, objectness, pred_circle

# TODO: COMPLETAR METODO
def dist_nms(outputs, anchors, masks):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        # t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    boxes = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    # class_probs = tf.concat(t, axis=1)

    # scores = confidence * class_probs
    scores = confidence
    # boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    #     boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 3)),
    #     scores=tf.reshape(
    #         scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
    #     max_output_size_per_class=FLAGS.yolo_max_boxes,
    #     max_total_size=FLAGS.yolo_max_boxes,
    #     iou_threshold=FLAGS.yolo_iou_threshold,
    #     score_threshold=FLAGS.yolo_score_threshold
    # )

    return boxes, scores
    # return boxes, scores, valid_detections


def DistOutput(filters, anchors, name=None):
    def dist_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        # Change the network to detec 2 coordenates for center, 1 for radius and
        # 1 for objectness instead of classes + center + width + heigth + obj
        # x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        # x = Lambda(lambda x:tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
        #                                   anchors, classes + 5)))(x)
        x = DarknetConv(x, anchors * (4), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, 4)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return dist_output


def DistNNTiny(size=None, channels=3, anchors=distNN_tiny_anchors,
               masks=distNN_tiny_anchor_masks, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='distNN_conv_0')(x)
    output_0 = DistOutput(256, len(masks[0]), name='distNN_output_0')(x)

    x = YoloConvTiny(128, name='distNN_conv_1')((x, x_8))
    output_1 = DistOutput(128, len(masks[1]), name='distNN_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: dist_circles(x, anchors[masks[0]]),
                     name='dist_circles_0')(output_0)
    boxes_1 = Lambda(lambda x: dist_circles(x, anchors[masks[1]]),
                     name='dist_circles_1')(output_1)
    outputs = Lambda(lambda x: dist_nms(x, anchors, masks),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')
    # return Model(inputs, (boxes_0, boxes_1), name='yolov3_tiny')


def DistLoss(anchors, ignore_thresh=0.5):
    def dist_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, r, obj))
        pred_circle, pred_obj, pred_xyr = dist_circles(
            y_pred, anchors)
        pred_xy = pred_xyr[..., 0:2]
        pred_r = pred_xyr[..., 2:3]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, r, obj)
        true_circle, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_circle[..., 0:2])
        true_radius = true_circle[..., 2:3]

        # give higher weights to small boxes
        circle_loss_scale = 2 - true_radius[..., 0] * true_radius[..., 0]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                  tf.cast(grid, tf.float32)
        true_radius = tf.math.log(true_radius / anchors)
        true_radius = tf.where(tf.math.is_inf(true_radius),
                               tf.zeros_like(true_radius), true_radius)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        # best_iou = tf.map_fn(
        #     lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
        #         x[1], tf.cast(x[2], tf.bool))), axis=-1),
        #     (pred_box, true_circle, obj_mask),
        #     tf.float32)
        ignore_mask = tf.cast(obj_mask == 0, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * circle_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        r_loss = obj_mask * circle_loss_scale * \
                 tf.reduce_sum(tf.square(true_radius - pred_r), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)

        # NO IGNORA LOS OBJETOS CON ALTO IOU
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        r_loss = tf.reduce_sum(r_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        return xy_loss + r_loss + obj_loss
        # return xy_loss + wh_loss + obj_loss

    return dist_loss
