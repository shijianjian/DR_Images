import tensorflow as tf

class TFRecordsUtils():

    IMAGE_FEATURE_NAME = 'image'
    LABEL_FEATURE_NAME = 'label'

    def _int64_feature(self, value):
        return tf.train.Feature(
                int64_list=tf.train.Int64List(value=value)
            )
    def _floats_feature(self, value):
        return tf.train.Feature(
                float_list=tf.train.FloatList(value=value)
            )

    def _bytes_feature(self, value):
        return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=value)
            )

    def _set_feature(self, image, label):
        return {
            self.LABEL_FEATURE_NAME: self._int64_feature([label]),
            self.IMAGE_FEATURE_NAME: self._bytes_feature(
                    # Note the square brackets here, to be a 1D list
                        [tf.compat.as_bytes(image.tostring())]
                    )
        }
    
    def _get_feature(self):
        return {
            self.IMAGE_FEATURE_NAME: tf.FixedLenFeature([1], tf.string),
            self.LABEL_FEATURE_NAME: tf.FixedLenFeature([1], tf.int64)
            }

    def _feature_retrieval(self, serialized_example, IMAGE_HEIGHT=3000, IMAGE_WIDTH=3000):
    
        features = tf.parse_single_example(
                    serialized_example,
                    features=self._get_feature()
                )
        # Decoding ...
        _image_raw = tf.cast(features[self.IMAGE_FEATURE_NAME], tf.string)[0]
        
        # Image processing
        image_shape = tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        # has to be uint8 type
        image_raw = tf.decode_raw(_image_raw, tf.uint8)
        image = tf.reshape(image_raw, image_shape)
        # Important, since we do not know the image shape information, it is all encoded in Tensorflow.
        # Hence, it can hardly pass the shape check in your Tensorflow Neural Network.
        resized_image = tf.image.resize_image_with_crop_or_pad(
            image=image,
            target_height=IMAGE_HEIGHT,
            target_width=IMAGE_WIDTH
        )
        resized_image = tf.cast(resized_image, tf.float32)
        # Label processing
        label = tf.cast(features[self.LABEL_FEATURE_NAME][0], tf.int64)
        
        return resized_image, label

    def read_TFRecords(self, tfrecords_filepath):
        items = []
        labels = []
        print("Loading %s" % tfrecords_filepath)
        for serialized_example in tf.python_io.tf_record_iterator(tfrecords_filepath):
            data, label = self._feature_retrieval(serialized_example)
            items.append(data)
            labels.append(label)
        print("Finished Loading %s" % tfrecords_filepath)
        return (items, labels)

    def load_TFRecords(self, filename_queue):
        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        
        image, label = self._feature_retrieval(serialized_example)
        
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=2,
            capacity=50,
            num_threads=1,
            min_after_dequeue=10)

        return images, labels

    def to_TFRecords(self, images, labels, output_file_path='data.tfrecords'):
        images = list(images)
        labels = list(labels)
        assert len(images) == len(labels), 'Must have equivalent number of images and labels.'
        print('Convert tfrecord file to %s' % output_file_path)
        with tf.python_io.TFRecordWriter(output_file_path) as writer:
                for i in range(len(images)):
                    example = tf.train.Example(
                    features=tf.train.Features(
                        feature = self._set_feature(
                                    images[i],
                                    labels[i])
                    ))
                    writer.write(example.SerializeToString())
                    print('\r{:.1%}'.format((i+1)/len(images)), end='')


if __name__ == '__main__':

    images, labels = TFRecordsUtils().read_TFRecords('../data.tfrecords')
    with tf.Session() as sess:
        import numpy as np
        print(np.array(images[0].eval()).shape)
        print(labels[0].eval())