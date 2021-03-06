import cv2
import numpy as np
import tensorflow as tf

class ImageUtils():

    @staticmethod
    def auto_crop_and_resize(image, resize_dims=False):
        '''
        Crop the major content from the image and resize to a wanted size.
        
        @param image
            The input image
            
        @param resize_dims
            To keep the origin shape, set it to False.
            Or give the specific shape of the wanted output.
        '''
        # Canny edge detector
        image_edges = cv2.Canny(image, 20, 30)
        # find the proper fully sized edges
        start_y, end_y = self._find_start_end_indices(image_edges, axis=0)
        start_x, end_x = self._find_start_end_indices(image_edges, axis=1)
        # crop
        image = image[start_x:end_x, start_y:end_y, :]
        if resize_dims is not False:
            # resize
            image = cv2.resize(image, resize_dims)
        return image

    @staticmethod
    def _find_start_end_indices(image, axis, threshold=10):
        '''
        Filter out the non-zero indices which means to carry contents.
        @param image
        @param axis
        @param threshold
            Ignore a specific number of pixels for each column.
            If the column carries less than threshold pixel numbers,
            it will be filtered.
        '''
        assert len(image.shape) == 2, "Must input a grayscaled image."
        hist_along_axis = np.count_nonzero(image, axis=axis)
        # plt.bar(range(hist_along_axis.shape[0]), hist_along_axis)
        idx_x = np.argwhere(hist_along_axis > threshold)
        return (idx_x[0][0], idx_x[-1][0])

    @staticmethod
    def mirror_image(img):
        return cv2.flip(img, 1)

    @staticmethod
    def compress_image(img):
        with tf.Session() as sess:
            res = sess.run(tf.image.encode_jpeg(img, quality=100))
        tf.contrib.keras.backend.clear_session()
        return res

    @staticmethod
    def decode_image(cpresd_img):
        with tf.Session() as sess:
            res = sess.run(tf.image.decode_image(cpresd_img))
        tf.contrib.keras.backend.clear_session()
        return res