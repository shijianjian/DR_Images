import tensorflow as tf
import os
from data_preparation.tfrecord_utils import TFRecordsUtils

class Trainer():

    def __init__(self):
        self.placeholders = []
        self.ops = {}

    def _get_model(self, image, output_units):
        # 3000 x 3000 x 3
        conv1 = tf.layers.conv2d(image, 32, 17, (10, 10), activation=tf.nn.relu)
        # 300 x 300 x 32
        pool1 = tf.layers.max_pooling2d(conv1, (5, 5), (5, 5))
        # 60 x 60 x 32
        conv2 = tf.layers.conv2d(pool1, 64, (5, 5), (1, 1), activation=tf.nn.relu)
        # 60 x 60 x 64
        pool2 = tf.layers.max_pooling2d(conv2, (5, 5), (3, 3))
        # 20 x 20 x 64
        conv3 = tf.layers.conv2d(pool2, 128, (5, 5), (1, 1), activation=tf.nn.relu)
        # 20 x 20 x 128
        pool3 = tf.layers.max_pooling2d(conv3, (5, 5), (2, 2))
        # 10 x 10 x 128
        dropout = tf.layers.dropout(pool3)
        flattern = tf.layers.flatten(dropout)
        # 10 x 10 x 128
        dense1 = tf.layers.dense(flattern, output_units, activation=tf.nn.softmax)
        
        return dense1

    def init_trainer(self, images_batch, labels_batch, output_units, learning_rate=0.1):
        # Get model and loss 
        pred = self._get_model(images_batch, output_units=output_units)

        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels_batch)
        loss = tf.reduce_mean(_loss)

        correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_batch))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
        self._add_op('pred', pred)
        self._add_op('loss', loss)
        self._add_op('accuracy', accuracy)
        self._add_op('train_op', train_op)

    def _get_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        return sess

    def _add_op(self, name, value):
        self.ops.update({name: value})

    def set_training_testing_files(self, training_files, testing_files=None):
        self.training_files = training_files
        self.testing_files = testing_files
        if self.testing_files == None:
            testing_files = []

    def make_data_handler(self, training_files, testing_files, batch_size=5):

        training_dataset = tf.data.TFRecordDataset(training_files) \
                                .map(TFRecordsUtils()._feature_retrieval) \
                                .batch(batch_size)

        testing_dataset = tf.data.TFRecordDataset(testing_files) \
                                .map(TFRecordsUtils()._feature_retrieval) \
                                .batch(batch_size)

        handle = tf.placeholder(tf.string, shape=[])

        iterator = tf.data.Iterator.from_string_handle(
            handle,
            training_dataset.output_types, 
            training_dataset.output_shapes
        )
        next_elem = iterator.get_next()

        train_init_iter = training_dataset.make_initializable_iterator()
        test_init_iter = testing_dataset.make_initializable_iterator()

        self._add_op('handle', handle)
        self._add_op('train_iter', train_init_iter)
        self._add_op('test_iter', test_init_iter)

        return next_elem

    def train(self, training_filenames, testing_filenames, batch_size=5, num_epochs=1):

        next_value = self.make_data_handler(training_filenames, testing_filenames, batch_size=batch_size)
        image_batch, labels_batch = next_value

        self.init_trainer(image_batch, labels_batch, 5)

        init_op = tf.group(
            tf.global_variables_initializer(),
            # if we have 'num_epochs' in filename_queue
            tf.local_variables_initializer()
        )

        with self._get_sess() as sess:
            sess.run(init_op)
            for e in range(num_epochs):
                print('Start training on epoch %d' % e)
                self.run_one_epoch(sess)
                print('Start testing on epoch %d' % e)
                self.run_one_epoch(sess, is_training=False)

    def run_one_epoch(self, sess, is_training=True):

        if is_training:
            training_handle = sess.run(self.ops['train_iter'].string_handle())
            sess.run(self.ops['train_iter'].initializer)
            while True:
                try:
                    _, _loss, _acc = sess.run(
                        [self.ops['train_op'], self.ops['loss'], self.ops['accuracy']],
                        feed_dict={
                            self.ops['handle']: training_handle
                            }
                    )
                    print(_acc, _loss, end='')
                except tf.errors.OutOfRangeError:
                    print('\n')
                    break
        else:
            testing_handle = sess.run(self.ops['test_iter'].string_handle())
            sess.run(self.ops['test_iter'].initializer)
            while True:
                try:
                    _loss, _acc = sess.run(
                        [self.ops['loss'], self.ops['accuracy']],
                        feed_dict={
                            self.ops['handle']: testing_handle
                            }
                    )
                    print(_acc, _loss, end='')
                except tf.errors.OutOfRangeError:
                    print('\n')
                    break 

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def train_test_files(path, test_ratio=0.2):

    for _, _, files in os.walk(path):
        count = len(files)
        files = [ os.path.join(path, x) for x in files]
        test_file_num = int(count*test_ratio)

        return files[:-test_file_num], files[-test_file_num:]



if __name__ == '__main__':

    training_filenames, testing_filenames = train_test_files('./data/tfrecords')

    print(get_available_gpus())
    with tf.device('/gpu:0'):
        trainer = Trainer()
        trainer.train(training_filenames[:1], testing_filenames[:1], batch_size=5, num_epochs=2)