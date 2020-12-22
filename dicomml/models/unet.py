from typing import Union

import tensorflow as tf

from dicomml.layers.unet import \
    UNETDownSampleBlock, UNETUpSampleBlock, UNETConvBlock


class UNETModel(tf.keras.Model):
    """
    Generalized UNET Model applied to all images
    in a case
    """

    def __init__(self,
                 depth: int = 4,
                 n_filters: int = 16,
                 downsampling_factor: int = 2,
                 dropoutrate: float = 0.1,
                 pool_three_dimensional: bool = False,
                 conv_three_dimensional: bool = False,
                 conv_block: Union[dict, None] = None,
                 conv_block_three_dimensional: bool = False,
                 **kwargs):
        super(UNETModel, self).__init__(**kwargs)
        conv_block = conv_block or {}
        if conv_block_three_dimensional:
            conv_block.update(dict(three_dimensional=True))
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

    def call(self, x):
        x = [x]
        for layer in self.downsampling_blocks:
            x = x + [*layer(x[-1])]
        y = self.middle_layer(x[-1])
        for layer in self.upsampling_blocks:
            y = layer(x.pop(), y)
        return y
