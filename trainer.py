import tensorflow as tf
import os
import numpy as np

class Trainer():

    def __init__(self, base_model='inception'):
        self.placeholders = []
        self.ops = {}
        self.base_model = base_model

        if self.base_model == 'inception':
            pass
        elif self.base_model == 'self_build':
            pass
        else:
            raise ValueError('Base model "%s" is not supported.')

    def _get_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        return sess

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

        init_op = tf.group(
                tf.global_variables_initializer(),
                # if we have 'num_epochs' in filename_queue
                tf.local_variables_initializer()
            )

        with self._get_sess() as sess:

            if self.base_model == 'self_build':
                from models.cnn import CNN

                cnn = CNN()

                training_filenames, testing_filenames = cnn.train_test_files(data_path)
                training_dataset, testing_dataset = cnn.make_dataset(training_filenames, testing_filenames, batch_size)
                next_value = cnn.make_data_handler(training_dataset, testing_dataset)
                image_batch, labels_batch = next_value

                cnn.init_trainer(image_batch, labels_batch, 5)

                saver = tf.train.Saver()

                sess.run(init_op)
                for e in range(num_epochs):
                    print('Start training on epoch %d' % e)
                    cnn.run_one_epoch(sess)

                    save_path = saver.save(sess, "./model_%d.ckpt" % e)
                    print("Model saved in path: %s" % save_path)

                    print('Start testing on epoch %d' % e)
                    cnn.run_one_epoch(sess, is_training=False)

            elif self.base_model == 'inception':
                from models.inception import Inception, EvaluateInputTensor
                inception = Inception()

                dims = (256, 256, 3)
                model = inception.init_model(dims, 5)
                training_gen, testing_gen = inception.set_data(data_path, target_size=dims[:2], batch_size=batch_size)
                
                # Bug fix for GPU mode
                from tensorflow import keras as K
                K.backend.get_session().run(tf.initialize_all_variables())
                model.fit_generator(
                    training_gen,
                    validation_data=testing_gen,
                    epochs=num_epochs,
                    callbacks=[EvaluateInputTensor(model, steps=100)],
                    verbose=1)

                import datetime
                model.save(os.path.abspath('model_%s.h5' % datetime.datetime.now().strftime('%m-%d-%H:%M:%S')))

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':

    # data_path = './data/tfrecords'
    data_path = './data/images_processed'
    num_epochs = 1
    batch_size = 16

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=data_path, help="Path to the data, default is '%s'" % data_path)
    parser.add_argument("--epochs", default=num_epochs, help="Number of epochs, default is '%d'" % num_epochs)
    parser.add_argument("--batch_size", default=batch_size, help="Batch size, default is '%d'" % batch_size)

    args = parser.parse_args()

    data_path = args.path
    batch_size = args.batch_size
    num_epochs = args.epochs

    print(get_available_gpus())
    with tf.device('/gpu:0'):
        trainer = Trainer()
        trainer.train(data_path, batch_size=batch_size, num_epochs=num_epochs)