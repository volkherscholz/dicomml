from typing import Type
from torch import nn


def slice_distributed(layer_class: Type[nn.Module]):

    class SliceDistributedLayer(layer_class):

        def forward(self, x, **kwargs):
            # assumes that x has the shape [batch, channels, slice, x, y]
            y = x.transpose(1, 2)
            # [batch, slice, channels, x, y]
            y_ = y.reshape([-1] + list(y.size())[2:5])
            # [batch * slice, channels, x, y]
            y_ = super(SliceDistributedLayer, self).forward(y_, **kwargs)
            # [batch * slice, out_channels, x, y]
            y = y_.reshape(list(y.size())[0:2] + list(y_.size())[1:4])
            # [batch, slice, out_channels, x, y]
            x = y.transpose(1, 2)
            # [batch, out_channels, slice, x, y]
            return x

    return SliceDistributedLayer


def image_distributed(layer_class: Type[nn.Module]):

    class ImageDistributedLayer(layer_class):

        def forward(self, x, **kwargs):
            # assumes that x has the shape [batch, channels, slice, x, y]
            y = x.reshape(list(x.size())[0:3] + [-1])
            # [batch, channels, slice, x * y]
            y = y.permute(0, 3, 1, 2)
            # [batch, x * y, channels, slice]
            y_ = y.reshape([-1] + list(y.size())[2:4])
            # [batch * x * y, channels, slice]
            y_ = super(ImageDistributedLayer, self).forward(y_, **kwargs)
            # [batch * x * y, out_channels, slice]
            y = y_.reshape(list(y.size())[0:2] + list(y_.size())[1:3])
            # [batch, x * y, out_channels, slice]
            y = y.permute(0, 2, 3, 1)
            # [batch, out_channels, slice, x * y]
            x = y.reshape(list(y.size())[0:3] + list(x.size())[3:5])
            # [batch, out_channels, slice, x, y]
            return x

    return ImageDistributedLayer
