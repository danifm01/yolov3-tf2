import pandas as pd
import tensorflow as tf
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import tqdm
from io import BytesIO
from PIL import Image

flags.DEFINE_string('output_file', 'data\\airsim_train_full.tfrecord', 'output '
                                                                  'dataset')


def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()


def build_example(param: pd.Series, image: str):
    img_path = os.path.join('data', 'airsim', 'Scene', image)
    # img_raw = open(img_path, 'rb').read()
    img_raw = Image.open(img_path)
    img_raw = convertToJpeg(img_raw)
    width = int(1920)
    height = int(1080)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    xmin.append(float(param['x1BB']) / width)
    ymin.append(float(param['y1BB']) / height)
    xmax.append(float(param['x2BB']) / width)
    ymax.append(float(param['y2BB']) / height)
    classes_text.append('Drone'.encode('utf8'))
    classes.append(int(0))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[width])),
        # 'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #     annotation['filename'].encode('utf8')])),
        # 'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #     annotation['filename'].encode('utf8')])),
        # 'image/key/sha256': tf.train.Feature(
        #     bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_raw])),
        # 'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(
            float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(
            float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(
            float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(
            float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=classes)),
        # 'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        # 'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def main(_argv):
    data = pd.read_csv('data\\airsim\\parametros.csv')
    data = data.set_index('nombre')
    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    path = os.path.join('data', 'airsim', 'Scene')
    image_list = os.listdir(path)
    print("Image list loaded:", len(image_list))
    for image in tqdm(image_list):
        tf_example = build_example(data.loc[image.replace('.png', '')], image)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Done')


if __name__ == '__main__':
    app.run(main)
