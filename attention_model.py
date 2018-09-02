# from keras.applications.vgg16 import VGG16 as PTModel
# from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
from keras.layers import BatchNormalization
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

def get_model(X, y):

    in_lay = Input(X.shape[1:])
    base_pretrained_model = PTModel(input_shape =  X.shape[1:], include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = False
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model(in_lay)
    bn_features = BatchNormalization()(pt_features)

    # here we do an attention mechanism to turn pixels in the GAP on an off

    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(1, 
                        kernel_size = (1,1), 
                        padding = 'valid', 
                        activation = 'sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
    out_layer = Dense(y.shape[-1], activation = 'softmax')(dr_steps)
    retina_model = Model(inputs = [in_lay], outputs = [out_layer])

    def top_2_accuracy(in_gt, in_pred):
        return top_k_categorical_accuracy(in_gt, in_pred, k=2)

    retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                            metrics = ['categorical_accuracy', top_2_accuracy])

    return retina_model

def get_callbacks():
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
    weight_path="{}_weights.best.hdf5".format('retina')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                                save_best_only=True, mode='min', save_weights_only = True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
    early = EarlyStopping(monitor="val_loss", 
                        mode="min", 
                        patience=6) # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [checkpoint, early, reduceLROnPlat]
    return callbacks_list

def get_data(path):
    datagen = ImageDataGenerator()
    training_gen = datagen.flow_from_directory(
        path,
        target_size=(256, 256),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1,
        save_to_dir=None
        )
    return training_gen

if __name__ == '__main__':

    data_gen = get_data('./data/images_processed')
    X, y = next(data_gen)
    retina_model = get_model(X, y)
    callbacks_list = get_callbacks()
    retina_model.fit_generator(
        data_gen, 
        steps_per_epoch = len(data_gen)// 1,
        epochs = 5, 
        callbacks = callbacks_list,
        workers = 0, # tf-generators are not thread-safe
        use_multiprocessing=False, 
        max_queue_size = 0
        )
    import os
    retina_model.save(os.path.join('model.h5'))