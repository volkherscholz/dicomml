from typing import Union


import tensorflow as tf


from dicomml.layers.wrappers import SliceDistributed, ImageDistributed


class UNETConvBlock(tf.keras.Layer):
    """
    Two convolutional layers with batch normalization
    """

    def __init__(self,
                 n_filters: int = 16,
                 kernel_size: int = 3,
                 depth: int = 2,
                 activation: str = 'relu',
                 kernel_initializer: str = 'he_normal',
                 batch_normalization: bool = True,
                 three_dimensional: bool = False,
                 conv_slice_direction: Union[dict, None] = None,
                 **kwargs):
        super(UNETConvBlock, self).__init__(**kwargs)
        if three_dimensional:
            self.layers = [tf.keras.layers.Conv3D(
                filters=n_filters,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding='same',
                kernel_initializer=kernel_initializer)
                for _ in range(depth)]
        else:
            self.layers = [SliceDistributed(tf.keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(kernel_size, kernel_size),
                padding='same',
                kernel_initializer=kernel_initializer))
                for _ in range(depth)]
        if conv_slice_direction is not None:
            self.layers.append(ImageDistributed(
                tf.keras.layers.Conv1D(**conv_slice_direction)))
        if batch_normalization:
            self.layers.append(
                SliceDistributed(tf.keras.layers.BatchNormalization()))
        self.layers.append(
            SliceDistributed(tf.keras.layers.Activation(activation)))

    def call(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)


class UNETDownSampleBlock(tf.keras.Layer):
    """
    Downsampling block: conv net followed by max pool
    """

    def __init__(self,
                 n_filters: int = 16,
                 pool_size: int = 2,
                 dropoutrate: float = 0.1,
                 pool_three_dimensional: bool = False,
                 conv_block: dict = dict(),
                 **kwargs):
        super(UNETDownSampleBlock, self).__init__(**kwargs)
        conv_block.update(dict(n_filters=n_filters))
        self.conv_layers = [
            UNETConvBlock(**conv_block),
            UNETConvBlock(**conv_block),
            SliceDistributed(tf.keras.layers.Dropout(dropoutrate))]
        if pool_three_dimensional:
            self.downsample_layer = tf.keras.layers.MaxPool3D(
                pool_size=(pool_size, pool_size, pool_size))
        else:
            self.downsample_layer = SliceDistributed(tf.keras.layers.MaxPool2D(
                pool_size=(pool_size, pool_size)))

    def call(self, x, **kwargs):
        for layer in self.conv_layers:
            x = layer(x, **kwargs)
        return x, self.downsample_layer(x)


class UNETUpSampleBlock(tf.keras.Layer):
    """
    Upsampling block: conv net followed by conv transpose
    """

    def __init__(self,
                 n_filters: int = 16,
                 stride_size: int = 2,
                 dropoutrate: float = 0.1,
                 conv_block: dict = dict(),
                 conv_three_dimensional: bool = False,
                 **kwargs):
        super(UNETUpSampleBlock, self).__init__(**kwargs)
        conv_block.update(dict(n_filters=n_filters))
        kernel_size = conv_block.get('kernel_size', 3)
        kernel_initializer = conv_block.get('kernel_initializer', 'he_normal')
        self.conv_layers = [
            SliceDistributed(tf.keras.layers.Dropout(dropoutrate)),
            UNETConvBlock(**conv_block),
            UNETConvBlock(**conv_block)]
        if conv_three_dimensional:
            self.upsample_layer = tf.keras.layers.Conv3DTranspose(
                    filters=n_filters,
                    kernel_size=(kernel_size, kernel_size, kernel_size),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    strides=(stride_size, stride_size, stride_size))
        else:
            self.upsample_layer = SliceDistributed(
                tf.keras.layers.Conv2DTranspose(
                    filters=n_filters,
                    kernel_size=(kernel_size, kernel_size),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    strides=(stride_size, stride_size)))

    def call(self, x, y, **kwargs):
        x = tf.concat([x, self.upsample_layer(y)], axis=-1)
        for layer in self.conv_layers:
            x = layer(x, **kwargs)
        return x
