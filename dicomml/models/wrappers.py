import tensorflow as tf


class SliceDistributed(tf.keras.layers.Wrapper):
    """
    Take a layer acting 2d images and let it act
    on all slices
    """

    def build(self, input_shape):
        # _input_shape = self._compute_in_shape(input_shape)
        _input_shape = input_shape
        # _input_shape[1] = None
        super(SliceDistributed, self).build(_input_shape[1:])

    def _compute_in_shape(self, input_shape):
        return tf.concat(
            [[input_shape[0] * input_shape[1]], input_shape[2:]], axis=0)
        # tuple([-1] + list(input_shape[2:]))

    def _compute_out_shape(self, features_shape, input_shape):
        return tf.concat(
            [[input_shape[0], input_shape[1]], features_shape[1:]], axis=0)
        # tuple([-1, input_shape[1]] + list(features_shape[1:]))

    def call(self, inputs, **kwargs):
        reshapes_inputs = tf.reshape(
            tensor=inputs,
            shape=self._compute_in_shape(tf.shape(inputs)))
        features = self.layer(reshapes_inputs)
        # reshape back
        return tf.reshape(
            tensor=features,
            shape=self._compute_out_shape(
                tf.shape(features), tf.shape(inputs)))


class ImageDistributed(tf.keras.layers.Wrapper):
    """
    Take a layer acting [batch, slice, channel]
    and distributed it to act on each generalized pixel
    of the images
    """

    def build(self, input_shape):
        # _input_shape = self._compute_in_shape(input_shape)
        _input_shape = [
            input_shape[0],  # * input_shape[1] * input_shape[2],
            input_shape[1],
            input_shape[-1]]
        # _input_shape[1] = None
        super(ImageDistributed, self).build(_input_shape)

    def _compute_in_shape(self, input_shape):
        return [input_shape[0] * input_shape[1] * input_shape[2],
                input_shape[3],
                input_shape[4]]

    def _compute_out_shape(self, features_shape, input_shape):
        return [input_shape[0],
                input_shape[1],
                input_shape[2],
                features_shape[1],
                features_shape[2]]

    def call(self, inputs, **kwargs):
        # in: [batch, slice, height, width, channels]
        # out: [batch, height, width, slice, channels]
        permuted_inputs = tf.transpose(
            inputs,
            perm=[0, 2, 3, 1, 4])
        reshapes_inputs = tf.reshape(
            tensor=permuted_inputs,
            shape=self._compute_in_shape(tf.shape(permuted_inputs)))
        # in: [batch * height * width, slice, channels]
        # out: [batch, height, width, feat, channels]
        features = self.layer(reshapes_inputs)
        # reshape back
        features_reshaped = tf.reshape(
            tensor=features,
            shape=self._compute_out_shape(
                tf.shape(features), tf.shape(permuted_inputs)))
        # in: [batch, height, width, feat, channels]
        # out: [batch, feat, height, width, channels]
        features_perm = tf.transpose(
            features_reshaped,
            perm=[0, 3, 1, 2, 4])
        return features_perm
