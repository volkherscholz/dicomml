import numpy as np
import tensorflow as tf
#
from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class CNNEncode(DicommlTransform):
    """
    Encode the images by running them through a CNN
    (mainly for transfer learning)
    """

    def __init__(self,
                 image_height=768,
                 image_width=768,
                 normalize_images=True,
                 n_images_batch=20,
                 cnn='inception'):
        _input_shape = image_height, image_width, 3
        if cnn != 'inception':
            raise ValueError('Only inception supported so far')
        cnn = tf.keras.applications.InceptionV3
        self.cnn = cnn(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=_input_shape,
            pooling=None
        )
        self.cnn.trainable = False
        self.normalize_images = normalize_images
        self.n_images_batch = n_images_batch

    def transform_case(self, case: DicommlCase) -> DicommlCase:
        index_to_arr_i = {}
        arrs = []
        for i, (index, arr) in enumerate(sorted(case.images.items())):
            if len(arr.shape) == 2:
                arr = arr[..., np.newaxis]
            arrs.append(arr)
            index_to_arr_i.update({index: i})
        imgarrays = np.array_split(
            np.stack(arrs, axis=0),
            indices_or_sections=self.n_images_batch,
            axis=0)
        _out = []
        for imgarray in imgarrays:
            _out.append(self._apply_cnn(imgarray))
        transformed_array = np.concatenate(_out, axis=0)
        # turn into dictionary again
        case.images = {
            index: transformed_array[i, ...]
            for index, i in index_to_arr_i.items()
        }
        return case

    def _apply_cnn(self, arr):
        if self.normalize_images:
            arr = tf.cast(arr, dtype=tf.float32)
            arr = tf.convert_to_tensor(arr, dtype=tf.float32) / 127.5
            arr -= 1.
        arr = tf.tile(arr, [1, 1, 1, 3])
        return tf.make_ndarray(self.cnn(arr))
