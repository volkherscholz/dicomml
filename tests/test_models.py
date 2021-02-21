import unittest

from torch import nn, randn

from dicomml.models.wrappers import slice_distributed, image_distributed
from dicomml.models.unet import \
    UNETConv, UNETConvBlock, UNETDownSampleBlock, UNETUpSampleBlock, UNETModel


class TestWrappers(unittest.TestCase):

    def test_slice_distributed(self):
        layer = slice_distributed(nn.Conv2d)(1, 1, 3, padding=1)
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape, data.shape)

    def test_image_distributed(self):
        layer = image_distributed(nn.Conv1d)(1, 1, 3, padding=1)
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape, data.shape)


class TestUNETConv(unittest.TestCase):

    def test_call_2d(self):
        layer = UNETConv()
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])

    def test_call2d1d(self):
        layer = UNETConv(conv_slice_direction=True)
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])

    def test_call_3d(self):
        layer = UNETConv(conv_three_dimensional=True)
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])


class TestUNETConvBlock(unittest.TestCase):

    def test_call_2d(self):
        layer = UNETConvBlock()
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])

    def test_call2d1d(self):
        layer = UNETConvBlock(conv_slice_direction=True)
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])

    def test_call_3d(self):
        layer = UNETConvBlock(conv_three_dimensional=True)
        data = randn(5, 1, 10, 120, 120)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])


class TestUNETDownSampleBlock(unittest.TestCase):

    def test_call_2d(self):
        layer = UNETDownSampleBlock()
        data = randn(5, 1, 10, 120, 120)
        output_1, output_2 = layer(data)
        self.assertEqual(output_1.shape[0], data.shape[0])
        self.assertEqual(output_2.shape[0], data.shape[0])

    def test_call_3d(self):
        layer = UNETDownSampleBlock(sample_three_dimensional=True)
        data = randn(5, 1, 10, 120, 120)
        output_1, output_2 = layer(data)
        self.assertEqual(output_1.shape[0], data.shape[0])
        self.assertEqual(output_2.shape[0], data.shape[0])


class TestUNETUpSampleBlock(unittest.TestCase):

    def test_call_2d(self):
        layer = UNETUpSampleBlock()
        x = randn(5, 1, 10, 120, 120)
        y = randn(5, 1, 10, 60, 60)
        output = layer(x, y)
        self.assertEqual(output.shape[0], x.shape[0])

    def test_call_3d(self):
        layer = UNETUpSampleBlock(sample_three_dimensional=True)
        x = randn(5, 1, 10, 120, 120)
        y = randn(5, 1, 5, 60, 60)
        output = layer(x, y)
        self.assertEqual(output.shape[0], x.shape[0])

    def test_call_2d_conv_upsample(self):
        layer = UNETUpSampleBlock(upsample_with_conv=True)
        x = randn(5, 1, 10, 120, 120)
        y = randn(5, 1, 10, 60, 60)
        output = layer(x, y)
        self.assertEqual(output.shape[0], x.shape[0])

    def test_call_3d_conv_upsample(self):
        layer = UNETUpSampleBlock(
            sample_three_dimensional=True, upsample_with_conv=True)
        x = randn(5, 1, 10, 120, 120)
        y = randn(5, 1, 5, 60, 60)
        output = layer(x, y)
        self.assertEqual(output.shape[0], x.shape[0])


class TestUNET(unittest.TestCase):

    def test_call_2d(self):
        model = UNETModel()
        images = randn(5, 1, 10, 120, 120)
        output = model(images)
        self.assertEqual(images.shape, output.shape)

    def test_call_2d_conv_upsample(self):
        model = UNETModel(upsample_with_conv=True)
        images = randn(5, 1, 10, 120, 120)
        output = model(images)
        self.assertEqual(images.shape, output.shape)

    def test_call_2d1d(self):
        model = UNETModel(conv_slice_direction=True)
        images = randn(5, 1, 10, 120, 120)
        output = model(images)
        self.assertEqual(images.shape, output.shape)

    def test_call_3d(self):
        model = UNETModel(
            conv_three_dimensional=True,
            sample_three_dimensional=True)
        images = randn(5, 1, 40, 120, 120)
        output = model(images)
        self.assertEqual(images.shape, output.shape)

    def test_call_3d_nodepthsampling(self):
        model = UNETModel(
            conv_three_dimensional=True,
            sample_three_dimensional=True,
            upsample_with_conv=True,
            sample_rate=[1, 2, 2],
            kernel_size=[1, 3, 3],
            padding=[0, 1, 1],
            block_depth=3)
        images = randn(5, 1, 10, 128, 128)
        output = model(images)
        self.assertEqual(images.shape, output.shape)
