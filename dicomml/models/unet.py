from torch import nn, sigmoid

from dicomml.models.wrappers import slice_distributed, image_distributed


class UNETConvBlock(nn.Module):
    """
    Two convolutional layers with batch normalization
    """

    def __init__(self,
                 n_channels_in: int = 1,
                 n_channels_out: int = 16,
                 kernel_size: int = 3,
                 depth: int = 2,
                 activation: bool = True,
                 batch_normalization: bool = True,
                 conv_three_dimensional: bool = False,
                 conv_slice_direction: bool = False,
                 **kwargs):
        super(UNETConvBlock, self).__init__()
        if conv_three_dimensional:
            self.layers = [
                nn.Conv3d(
                    in_channels=n_channels_in,
                    out_channels=n_channels_out,
                    kernel_size=kernel_size,
                    padding=1)] + [
                nn.Conv3d(
                    in_channels=n_channels_out,
                    out_channels=n_channels_out,
                    kernel_size=kernel_size,
                    padding=1)
                for _ in range(depth - 1)]
        else:
            self.layers = [
                slice_distributed(nn.Conv2d)(
                    in_channels=n_channels_in,
                    out_channels=n_channels_out,
                    kernel_size=kernel_size,
                    padding=1)] + [
                slice_distributed(nn.Conv2d)(
                    in_channels=n_channels_out,
                    out_channels=n_channels_out,
                    kernel_size=kernel_size,
                    padding=1)
                for _ in range(depth - 1)]
        if conv_slice_direction:
            self.layers.append(image_distributed(
                nn.Conv1d)(
                    in_channels=n_channels_out,
                    out_channels=n_channels_out,
                    kernel_size=kernel_size,
                    padding=1))
        if batch_normalization:
            self.layers.append(
                slice_distributed(nn.BatchNorm2d)(num_features=n_channels_out))
        if activation:
            self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class UNETDownSampleBlock(UNETConvBlock):
    """
    Downsampling block: conv net followed by max pool
    """

    def __init__(self,
                 sample_rate: int = 2,
                 dropoutrate: float = 0.1,
                 sample_three_dimensional: bool = False,
                 **kwargs):
        super(UNETDownSampleBlock, self).__init__(**kwargs)
        self.dropout = slice_distributed(nn.Dropout2d)(p=dropoutrate)
        if sample_three_dimensional:
            self.downsample_layer = nn.MaxPool3d(kernel_size=sample_rate)
        else:
            self.downsample_layer = slice_distributed(
                nn.MaxPool2d)(kernel_size=sample_rate)

    def forward(self, x, **kwargs):
        x = self.dropout(super(UNETDownSampleBlock, self).forward(x))
        return x, self.downsample_layer(x)


class UNETUpSampleBlock(UNETConvBlock):
    """
    Upsampling block: conv net followed by conv transpose
    """

    def __init__(self,
                 sample_rate: int = 2,
                 dropoutrate: float = 0.1,
                 sample_three_dimensional: bool = False,
                 **kwargs):
        super(UNETUpSampleBlock, self).__init__(**kwargs)
        self.dropout = slice_distributed(nn.Dropout2d)(p=dropoutrate)
        if sample_three_dimensional:
            self.upsample_layer = \
                nn.Upsample(scale_factor=sample_rate, mode='trilinear')
        else:
            self.upsample_layer = slice_distributed(
                nn.Upsample)(scale_factor=sample_rate, mode='bilinear')

    def forward(self, x, y, **kwargs):
        x = x + self.upsample_layer(y)
        x = self.dropout(super(UNETUpSampleBlock, self).forward(x))
        return x


class UNETModel(nn.Module):
    """
    Generalized UNET Model applied to all images
    in a case
    """

    def __init__(self,
                 n_channels_in: int = 1,
                 n_filters: int = 16,
                 block_depth: int = 2,
                 n_classes: int = 1,
                 **kwargs):
        super(UNETModel, self).__init__()
        self.downsampling_blocks = nn.ModuleList([
            UNETDownSampleBlock(
                n_channels_in=1,
                n_channels_out=n_filters,
                **kwargs)] + [
            UNETDownSampleBlock(
                n_channels_in=2**i * n_filters,
                n_channels_out=2**(i + 1) * n_filters,
                **kwargs) for i in range(block_depth)])
        self.middle_layer = UNETConvBlock(
            n_channels_in=2**block_depth * n_filters,
            n_channels_out=2**block_depth * n_filters,
            **kwargs)
        self.upsampling_blocks = nn.ModuleList([
            UNETUpSampleBlock(
                n_channels_in=2**i * n_filters,
                n_channels_out=2**(i - 1) * n_filters,
                **kwargs) for i in range(block_depth, 0, -1)] + [
            UNETUpSampleBlock(
                n_channels_in=n_filters,
                n_channels_out=n_classes,
                **kwargs)])

    def forward(self, x):
        ys = []
        for layer in self.downsampling_blocks:
            y, x = layer(x)
            ys.append(y)
        x = self.middle_layer(x)
        for layer in self.upsampling_blocks:
            x = layer(ys.pop(), x)
        return sigmoid(x)
