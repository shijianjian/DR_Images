import cv2
import pandas as pd
import os
from data_preparation.image_utils import ImageUtils
from data_preparation.tfrecord_converter import TFRecordsConverter

def read_images(path, convert_func, func_args):
    imgs = {}
    path = os.path.abspath(path)
    print('Reading images in %s' % path)
    for _, _, files in os.walk(path):
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(path, f))
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if convert_func is not None:
                # image conversion
                converted_image = convert_func(orig_img, func_args)
                imgs.update({f[:f.rfind('.')]: converted_image})
            else:
                imgs.update({f[:f.rfind('.')]: orig_img})
            print('\r{:.1%}'.format((i+1)/len(files)), end='')
    print('\n')
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
    TFRecordsConverter().to_TFRecords(imgs_dict.values(), labels_dict.values())