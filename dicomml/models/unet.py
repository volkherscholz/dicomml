from typing import Union

import tensorflow as tf

from dicomml.models.wrappers import SliceDistributed, ImageDistributed


class UNETConvBlock(tf.keras.layers.Layer):
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
                 conv_three_dimensional: bool = False,
                 conv_slice_direction: bool = False,
                 **kwargs):
        super(UNETConvBlock, self).__init__(**kwargs)
        if conv_three_dimensional:
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
        if conv_slice_direction:
            self.layers.append(ImageDistributed(
                tf.keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size,
                    padding='same',
                    kernel_initializer=kernel_initializer)))
        if batch_normalization:
            self.layers.append(
                SliceDistributed(tf.keras.layers.BatchNormalization()))
        self.layers.append(
            SliceDistributed(tf.keras.layers.Activation(activation)))

    def call(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class UNETDownSampleBlock(tf.keras.layers.Layer):
    """
    Downsampling block: conv net followed by max pool
    """

    def __init__(self,
                 n_filters: int = 16,
                 pool_size: int = 2,
                 dropoutrate: float = 0.1,
                 pool_three_dimensional: bool = False,
                 conv_block: Union[dict, None] = None,
                 **kwargs):
        super(UNETDownSampleBlock, self).__init__(**kwargs)
        conv_block = conv_block or {}
        conv_block = {'n_filters': n_filters, **conv_block}
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


class UNETUpSampleBlock(tf.keras.layers.Layer):
    """
    Upsampling block: conv net followed by conv transpose
    """

    def __init__(self,
                 n_filters: int = 16,
                 stride_size: int = 2,
                 upsample_size: int = 2,
                 dropoutrate: float = 0.1,
                 conv_block: Union[dict, None] = None,
                 conv_three_dimensional: bool = False,
                 **kwargs):
        super(UNETUpSampleBlock, self).__init__(**kwargs)
        conv_block = conv_block or {}
        conv_block = {'n_filters': n_filters, **conv_block}
        kernel_initializer = conv_block.get('kernel_initializer', 'he_normal')
        self.conv_layers = [
            SliceDistributed(tf.keras.layers.Dropout(dropoutrate)),
            UNETConvBlock(**conv_block),
            UNETConvBlock(**conv_block)]
        if conv_three_dimensional:
            self.upsample_layer = tf.keras.layers.Conv3DTranspose(
                    filters=n_filters,
                    kernel_size=(upsample_size, upsample_size, upsample_size),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    strides=(stride_size, stride_size, stride_size))
        else:
            self.upsample_layer = SliceDistributed(
                tf.keras.layers.Conv2DTranspose(
                    filters=n_filters,
                    kernel_size=(upsample_size, upsample_size),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    strides=(stride_size, stride_size)))

    def call(self, x, y, **kwargs):
        x = tf.concat([x, self.upsample_layer(y)], axis=-1)
        for layer in self.conv_layers:
            x = layer(x, **kwargs)
        return x


class UNETModel(tf.keras.Model):
    """
    Generalized UNET Model applied to all images
    in a case
    """

    def __init__(self,
                 depth: int = 2,
                 n_channels: int = 1,
                 n_filters: int = 16,
                 downsampling_factor: int = 2,
                 dropoutrate: float = 0.1,
                 conv_block: Union[dict, None] = None,
                 pool_three_dimensional: bool = False,
                 conv_three_dimensional: bool = False,
                 conv_slice_direction: bool = False,
                 **kwargs):
        super(UNETModel, self).__init__(**kwargs)
        conv_block = conv_block or {}
        conv_block = {
            **conv_block,
            'conv_three_dimensional': conv_three_dimensional,
            'conv_slice_direction': conv_slice_direction}
        self.downsampling_blocks = [
            UNETDownSampleBlock(
                n_filters=int(n_filters * i),
                pool_size=downsampling_factor,
                dropoutrate=dropoutrate,
                conv_block=conv_block,
                pool_three_dimensional=pool_three_dimensional)
            for i in range(1, depth + 1)]
        self.middle_layer = UNETConvBlock(
            n_filters=int(n_filters * (depth + 1)),
            **conv_block)
        self.upsampling_blocks = [
            UNETUpSampleBlock(
                n_filters=int(n_filters * i),
                stride_size=downsampling_factor,
                dropoutrate=dropoutrate,
                conv_block=conv_block,
                conv_three_dimensional=conv_three_dimensional)
            for i in range(depth, 0, -1)]
        self.last_layer = SliceDistributed(tf.keras.layers.Conv2D(
            filters=n_channels,
            kernel_size=(1, 1),
            activation='sigmoid',
            padding='same'))

    def call(self, x):
        ys = []
        for layer in self.downsampling_blocks:
            y, x = layer(x)
            ys.append(y)
        x = self.middle_layer(x)
        for layer in self.upsampling_blocks:
            x = layer(ys.pop(), x)
        return self.last_layer(x)
