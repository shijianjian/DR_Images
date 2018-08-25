import tensorflow as tf
import os
import numpy as np
from data_preparation.tfrecord_utils import TFRecordsUtils
from tensorflow import keras as K

class EvaluateInputTensor(K.callbacks.Callback):
    # Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_tfrecord.py#L56
    """ Validate a model which does not expect external numpy data during training.
    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.
    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.
    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


class Trainer():

    def __init__(self, base_model='inception'):
        self.placeholders = []
        self.ops = {}
        self.base_model = base_model

        if self.base_model == 'inception':
            pass
        elif self.base_model == 'simple_built':
            pass
        else:
            raise ValueError('Base model "%s" is not supported.')
        

    def _get_model(self, image, n_classes):
        # 3000 x 3000 x 3
        conv1 = tf.layers.conv2d(image, 32, 17, (10, 10), activation=tf.nn.relu, name='conv_1')
        # 300 x 300 x 32
        pool1 = tf.layers.max_pooling2d(conv1, (5, 5), (5, 5), name='pool_1')
        # 60 x 60 x 32
        conv2 = tf.layers.conv2d(pool1, 64, (5, 5), (1, 1), activation=tf.nn.relu, name='conv_2')
        # 60 x 60 x 64
        pool2 = tf.layers.max_pooling2d(conv2, (5, 5), (3, 3), name='pool_2')
        # 20 x 20 x 64
        conv3 = tf.layers.conv2d(pool2, 128, (5, 5), (1, 1), activation=tf.nn.relu, name='conv_3')
        # 20 x 20 x 128
        pool3 = tf.layers.max_pooling2d(conv3, (5, 5), (2, 2), name='pool_3')
        # 10 x 10 x 128
        dropout = tf.layers.dropout(pool3, name='dropout')
        flattern = tf.layers.flatten(dropout, name='flattern')
        # 10 x 10 x 128
        dense1 = tf.layers.dense(flattern, n_classes, activation=tf.nn.softmax, name='dense_1')
        
        return dense1
    
    def init_inception_trainer(self, n_classes):
        # code snippet taken from https://keras.io/applications/
        base_model = K.applications.inception_v3.InceptionV3(input_tensor=None, input_shape=(3000, 3000, 3), weights='imagenet', include_top=False)

        output = base_model.output
        output = K.layers.GlobalAveragePooling2D()(output)
        # add a fully-connected layer
        output = K.layers.Dense(1024, activation='relu')(output)
        # and a logistic layer
        predictions = K.layers.Dense(n_classes, activation='softmax')(output)

        model = K.Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False
            
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy'
            )

        return model

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

    def make_data_handler(self, training_dataset, testing_dataset):

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
    
    def make_dataset(self, training_files, testing_files, batch_size=5):
        training_dataset = tf.data.TFRecordDataset(training_files) \
                                .map(TFRecordsUtils()._feature_retrieval) \
                                .batch(batch_size)

        testing_dataset = tf.data.TFRecordDataset(testing_files) \
                                .map(TFRecordsUtils()._feature_retrieval) \
                                .batch(batch_size)
        
        return training_dataset, testing_dataset

    def train(self, data_path, batch_size=5, num_epochs=1):
        '''
        data_path: 
            the path to tfrecords if self.base_model is set to 'self_built',
            otherwise using the pre-processed Keras image generator folder structure:
            which can be done by data_preparation.py.
            the proper structure shall be:
                - folder 
                    - class A
                    - class B
                    ...
        '''

        tf.reset_default_graph()

        if self.base_model == 'self_built':

            training_filenames, testing_filenames = train_test_files(data_path)
            training_dataset, testing_dataset = self.make_dataset(training_filenames, testing_filenames, batch_size)

            next_value = self.make_data_handler(training_dataset, testing_dataset)
            image_batch, labels_batch = next_value

            self.init_trainer(image_batch, labels_batch, 5)
            init_op = tf.group(
                tf.global_variables_initializer(),
                # if we have 'num_epochs' in filename_queue
                tf.local_variables_initializer()
            )
            saver = tf.train.Saver()

            with self._get_sess() as sess:
                sess.run(init_op)
                for e in range(num_epochs):
                    print('Start training on epoch %d' % e)
                    self.run_one_epoch(sess)

                    save_path = saver.save(sess, "./model_%d.ckpt" % e)
                    print("Model saved in path: %s" % save_path)

                    print('Start testing on epoch %d' % e)
                    self.run_one_epoch(sess, is_training=False)

        elif self.base_model == 'inception':

            model = self.init_inception_trainer(5)

            with self._get_sess() as sess:

                datagen = K.preprocessing.image.ImageDataGenerator()
                training_gen = datagen.flow_from_directory(
                    data_path,
                    target_size=(3000, 3000),
                    color_mode='rgb',
                    class_mode='categorical',
                    batch_size=batch_size,
                    save_to_dir=None,
                    interpolation=None
                    )

                model.fit_generator(
                    training_gen,
                    epochs=num_epochs,
                    callbacks=[EvaluateInputTensor(model, steps=100)],
                    verbose=1)

                model.save(os.path.abspath('.'))
                


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
                    print('\racc: {:.1%}, loss: {:.5}'.format(_acc, _loss), end='')
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
                    print('\racc: {:.1%}, loss: {:.5}'.format(_acc, _loss), end='')
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

    data_path = './data/tfrecords'
    data_path = './data/images_processed'

    print(get_available_gpus())
    with tf.device('/gpu:0'):
        trainer = Trainer()
        trainer.train(data_path, batch_size=5, num_epochs=2)