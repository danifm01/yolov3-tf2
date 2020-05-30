from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)


def main():
    model = YoloV3Tiny(416, training=True)
    model.summary()


if __name__ == '__main__':
    main()
