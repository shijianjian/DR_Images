import cv2
import pandas as pd
import os
import tensorflow as tf
from utils.image_utils import ImageUtils
from utils.tfrecord_utils import TFRecordsUtils

RESIZE_DIMS=(512, 512)

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
            imgs.update({f[:f.rfind('.')]: ImageUtils.compress_image(img)})
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

def to_keras_generator_folder_structure(images_path, labels_path):
    labels = read_labels(labels_path)

    import os

    images_path = os.path.abspath(images_path)

    out_path = os.path.abspath(images_path + "_processed")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for _, _, files in os.walk(images_path):
        for i, f in enumerate(files):

            value = str(labels[f[:f.rfind('.')]])
            if not os.path.exists(os.path.join(out_path, value)):
                os.makedirs(os.path.join(out_path, value))
            
            # shutil.move(os.path.join(images_path, f), os.path.join(images_path, value, f))
            img = cv2.imread(os.path.join(images_path, f))
            img = ImageUtils.auto_crop_and_resize(img, resize_dims=RESIZE_DIMS)
            # Filpping the right eye image to the left representation
            if value.endswith('right'):
                img = ImageUtils.mirror_image(img)
            cv2.imwrite(os.path.join(out_path, value, f), img)

            print('\r{:.1%}'.format((i+1)/len(files)), end='')

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
        yield ImageUtils.compress_image(img), f[f.rfind('/') + 1:]

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
        
        data = streaming_images(files, ImageUtils.auto_crop_and_resize, RESIZE_DIMS)
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

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the data", required=True)
    parser.add_argument("--resize", help="Resize the images to a specific size, default is %s" % str(RESIZE_DIMS))
    parser.add_argument('--tfrecord', action='store_true')
    parser.add_argument('--keras', action='store_true')

    args = parser.parse_args()

    USING_KERAS = args.keras
    USING_TFRECORD = args.tfrecord
    PATH = args.path

    image_path = os.path.join(PATH, 'images')
    label_path = os.path.join(PATH, 'labels')
    if not os.path.isdir(image_path):
        raise ValueError("Image directory %s does not exist" % image_path)
    else:
        print("Image directory '%s' found" % image_path)
    if not os.path.isdir(label_path):
        raise ValueError("Label directory %s does not exist" % label_path)
    else:
        print("Label directory '%s' found" % label_path)

    if args.resize is not None:
        RESIZE_DIMS = eval(args.resize)
    print("Images will be resized to %s" % RESIZE_DIMS)
    
    if (USING_KERAS and USING_TFRECORD) or (not USING_KERAS and not USING_TFRECORD):
        raise ValueError("Flag --keras and --tfrecord can't be both true or false. Using only one instead.")

    if USING_TFRECORD:
        import numpy as np
        # imgs_dict = read_images('data/images', ImageUtils().auto_crop_and_resize, (3000, 3000))
        # labels_dict = read_labels('data/labels')
        # # Presume they are in the same order
        # TFRecordsUtils().to_TFRecords(imgs_dict.values(), labels_dict.values())

        transfer_to_tfrecords(image_path, label_path, trunk_size=100)
    elif USING_KERAS:
        to_keras_generator_folder_structure(image_path, label_path)

