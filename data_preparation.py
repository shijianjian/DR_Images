import cv2
import pandas as pd
import os
import tensorflow as tf
from data_preparation.image_utils import ImageUtils
from data_preparation.tfrecord_utils import TFRecordsUtils

def read_images(path, convert_func, func_args):
    imgs = {}
    path = os.path.abspath(path)
    print('Reading images in %s' % path)
    sess = tf.Session()
    for _, _, files in os.walk(path):
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(path, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if convert_func is not None:
                # image conversion
                img = convert_func(img, func_args)
            imgs.update({f[:f.rfind('.')]: ImageUtils().compress_image(sess, img)})
            print('\r{:.1%}'.format((i+1)/len(files)), end='')
    print('\n')
    sess.close()
    return imgs

def read_labels(path):
    labels = {}
    path = os.path.abspath(path)
    print('Reading labels in %s' % path)
    for _, _, files in os.walk(path):
        for f in files:
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, f))
                for i in range(len(df)):
                    labels.update({df['image'][i]:df['level'][i]})
                    print('\r{:.1%}'.format((i+1)/len(df)), end='')
                print('\n')
    return labels


image_utils = ImageUtils()
tf_records_utils = TFRecordsUtils()

def find_related_files(path, start_from=0, trunk_size=100):
    for _, _, files in os.walk(path):
        return [ os.path.join(path, x) for x in files[start_from : start_from + trunk_size] ]

def streaming_images(files, convert_func, func_args):
    for i, f in enumerate(files):
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if convert_func is not None:
            # image conversion
            img = convert_func(img, func_args)
        yield image_utils.compress_image(img), f[f.rfind('/') + 1:]

def save_image(writer, image, label):
    example = tf.train.Example(
        features=tf.train.Features(
            feature = tf_records_utils._set_feature(image, label)
        )
    )
    writer.write(example.SerializeToString())
    writer.flush()


def transfer_to_tfrecords(image_path, label_path, start_from=0, trunk_size=100):

    for _, _, files in os.walk(image_path):
        total_size = len(files)

    if total_size % trunk_size == 0:
        total_bunch = int(total_size/trunk_size)
    else:
        total_bunch = int(total_size/trunk_size) + 1

    labels = read_labels(label_path)

    for trunk_num in range(start_from, total_bunch):

        files = find_related_files(image_path, start_from=trunk_num*trunk_size, trunk_size=trunk_size)
        
        data = streaming_images(files, image_utils.auto_crop_and_resize, (3000, 3000))
        output_file_path = 'data_%d.tfrecords' % trunk_num

        writer = tf.python_io.TFRecordWriter(output_file_path)
        print('\nWriting to %s' % output_file_path)

        for i, (image, f) in enumerate(data):
            save_image(writer, image, labels[f[:f.rfind('.')]])
            if (i+1)/trunk_size - int((i+1)/trunk_size) == 0:
                import gc, time
                gc.collect()
                print('100%', end='')
                time.sleep(10)
            else:
                print('\r{:.1%}'.format((i+1)/trunk_size - int((i+1)/trunk_size)), end='')    
        
        writer.flush()    
        writer.close()
        print('Finished on %s' % output_file_path)


if __name__ == '__main__':
    import numpy as np
    # imgs_dict = read_images('data/images', ImageUtils().auto_crop_and_resize, (3000, 3000))
    # labels_dict = read_labels('data/labels')
    # # Presume they are in the same order
    # TFRecordsUtils().to_TFRecords(imgs_dict.values(), labels_dict.values())

    transfer_to_tfrecords('data/images', 'data/labels', trunk_size=100)
