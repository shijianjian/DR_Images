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
    
if __name__ == '__main__':
    
    imgs_dict = read_images('data/images', ImageUtils().auto_crop_and_resize, (3000, 3000))
    labels_dict = read_labels('data/labels')
    # Presume they are in the same order
    TFRecordsUtils().to_TFRecords(imgs_dict.values(), labels_dict.values())