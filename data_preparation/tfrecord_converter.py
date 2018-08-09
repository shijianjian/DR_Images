import tensorflow as tf

class TFRecordsConverter():

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

    def get_feature(self, image, label):
        return {
            'label': self._int64_feature([label]),
            'image': self._bytes_feature(
                    # Note the square brackets here, to be a 1D list
                        [tf.compat.as_bytes(image.tostring())]
                    )
        }
        
    def to_TFRecords(self, images, labels):
        images = list(images)
        labels = list(labels)
        # print(labels)
        assert len(images) == len(labels), 'Must have equivalent number of images and labels.'
        output_file_path = 'data.tfrecords'
        print('Convert tfrecord file to %s' % output_file_path)
        with tf.python_io.TFRecordWriter(output_file_path) as writer:
                for i in range(len(images)):
                    example = tf.train.Example(
                    features=tf.train.Features(
                        feature = self.get_feature(
                                    images[i],
                                    labels[i])
                    ))
                    writer.write(example.SerializeToString())
                    print('\r{:.1%}'.format((i+1)/len(images)), end='')