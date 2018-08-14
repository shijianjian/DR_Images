import tensorflow as tf
from data_preparation.tfrecord_utils import TFRecordsUtils

def _get_model(image, output_units):
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

def get_trainer(images_batch, labels_batch, output_units, learning_rate=0.1):
    # Get model and loss 
    pred = _get_model(images_batch, output_units=output_units)
    print(pred, labels_batch)
    _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels_batch)
    loss = tf.reduce_mean(_loss)

    correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_batch))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    ops = {
        'pred': pred,
        'loss': loss,
        'accuracy': accuracy,
        'train_op': train_op
    }
    return ops

def get_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    return sess

def train(num_epochs):

    tf.reset_default_graph()
    
#     filename_queue = tf.FIFOQueue(num_epochs, tf.string)
#     enqueue_placeholder = tf.placeholder(dtype=tf.string)
#     enqueue_op = filename_queue.enqueue(enqueue_placeholder)
    
#     images_batch, labels_batch = TFRecordsUtils().load_TFRecords(filename_queue)
    
    filename_queue = tf.train.string_input_producer(
            ['./data.tfrecords'], 
            num_epochs=num_epochs
        )
    images_batch, labels_batch = TFRecordsUtils().load_TFRecords(filename_queue)
    ops = get_trainer(images_batch, labels_batch, 5)
    
    init_op = tf.group(
        tf.global_variables_initializer(),
        # if we have 'num_epochs' in filename_queue
        tf.local_variables_initializer()
    )
    
    with get_sess() as sess:
        sess.run(init_op)
#         for i in range(num_epochs):
#             sess.run([enqueue_op], feed_dict={enqueue_placeholder:"../data.tfrecords"})
        train_one_epoch(sess, ops)
    
def train_one_epoch(sess, ops):
    # Start training
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    count = 0
    while not coord.should_stop():
        try:
            _, _loss, _acc = sess.run([ops['train_op'], ops['loss'], ops['accuracy']])
            count += 1
            print(_acc, _loss)
            if count == 5:
                # For test set
                pass
        except tf.errors.OutOfRangeError:
            print('Finished one epoch')
            break
            
    coord.request_stop()
    coord.join(threads)


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    print(get_available_gpus())
    with tf.device('/gpu:0'):
        train(10)