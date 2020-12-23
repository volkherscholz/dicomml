import unittest
import tensorflow as tf

from dicomml.models.wrappers import SliceDistributed, ImageDistributed
from dicomml.models.unet import \
    UNETConvBlock, UNETDownSampleBlock, UNETUpSampleBlock, UNETModel


class TestWrappers(unittest.TestCase):

    def test_slice_distributed(self):
        base_layer = tf.keras.layers.Conv2D(4, (3, 3))
        layer = SliceDistributed(base_layer)
        data = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])

    def test_image_distributed(self):
        base_layer = tf.keras.layers.Conv1D(4, 3)
        layer = ImageDistributed(base_layer)
        data = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])


class TestUNETConvBlock(unittest.TestCase):

    def test_call_2d(self):
        layer = UNETConvBlock()
        data = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])

    def test_call2d1d(self):
        layer = UNETConvBlock(conv_slice_direction=True)
        data = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])

    def test_call_3d(self):
        layer = UNETConvBlock(conv_three_dimensional=True)
        data = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output = layer(data)
        self.assertEqual(output.shape[0], data.shape[0])


class TestUNETDownSampleBlock(unittest.TestCase):

    def test_call_2d(self):
        layer = UNETDownSampleBlock()
        data = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output_1, output_2 = layer(data)
        self.assertEqual(output_1.shape[0], data.shape[0])
        self.assertEqual(output_2.shape[0], data.shape[0])

    def test_call_3d(self):
        layer = UNETDownSampleBlock(pool_three_dimensional=True)
        data = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output_1, output_2 = layer(data)
        self.assertEqual(output_1.shape[0], data.shape[0])
        self.assertEqual(output_2.shape[0], data.shape[0])


class TestUNETUpSampleBlock(unittest.TestCase):

    def test_call_2d(self):
        layer = UNETUpSampleBlock()
        x = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        y = tf.random.normal((5, 10, 60, 60, 1), mean=0.0, stddev=1.0)
        output = layer(x, y)
        self.assertEquals(output.shape[0], x.shape[0])

    def test_call_3d(self):
        layer = UNETUpSampleBlock(conv_three_dimensional=True)
        x = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        y = tf.random.normal((5, 5, 60, 60, 1), mean=0.0, stddev=1.0)
        output = layer(x, y)
        self.assertEqual(output.shape[0], x.shape[0])


class TestUNET(unittest.TestCase):

    def test_call_2d(self):
        model = UNETModel()
        images = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output = model(images)
        self.assertEqual(images.shape, output.shape)

    def test_call_2d1d(self):
        model = UNETModel(conv_slice_direction=True)
        images = tf.random.normal((5, 10, 120, 120, 1), mean=0.0, stddev=1.0)
        output = model(images)
        self.assertEqual(images.shape, output.shape)

    def test_call_3d(self):
        model = UNETModel(
            pool_three_dimensional=True, conv_three_dimensional=True)
        images = tf.random.normal((5, 20, 120, 120, 1), mean=0.0, stddev=1.0)
        output = model(images)
        self.assertEqual(images.shape, output.shape)
