from typing import List, Union

import numpy as np
from scipy import ndimage
from skimage import morphology, measure

from dicomml.transforms import ArrayTransform


class Shift(ArrayTransform):
    """
    Shift in x or y direction
    """

    def __init__(self,
                 x_shift: int = 0,
                 y_shift: int = 0,
                 **kwargs):
        super(Shift, self).__init__(**kwargs)
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _transform_array(self, array):
        array = np.roll(array, self.x_shift, axis=0)
        array = np.roll(array, self.y_shift, axis=1)
        return array


class Mirror(ArrayTransform):
    """
    Mirror along x or y direction
    """

    def __init__(self,
                 axis: int = -1,
                 **kwargs):
        super(Mirror, self).__init__(**kwargs)
        self.axis = axis

    def _transform_array(self, array):
        return np.flip(array, axis=self.axis)


class Cut(ArrayTransform):
    """
    Cut arrays
    """

    def __init__(self,
                 x_range: List[Union[int, None]] = [None, None],
                 y_range: List[Union[int, None]] = [None, None],
                 **kwargs):
        super(Cut, self).__init__(**kwargs)
        self.x_range = x_range
        self.y_range = y_range

    def _transform_array(self, array):
        return array[
            self.x_range[0]:self.x_range[1],
            self.y_range[0]:self.y_range[1]]


class Pad(ArrayTransform):
    """
    Pad arrays so that they have shape
    """

    def __init__(self,
                 target_shape: List[int],
                 fill_value: float = 0.0,
                 **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.target_shape = target_shape
        self.fill_value = fill_value

    def _transform_array(self, array):
        _shape = array.shape
        _array = self.fill_value * np.ones(self.target_shape)
        # try to center the array more or less
        _x = int((self.target_shape[0] - _shape[0]) // 2)
        _y = int((self.target_shape[1] - _shape[1]) // 2)
        _array[_x:(_x + _shape[0]), _y:(_y + _shape[1])] = array
        return _array


class Rotate(ArrayTransform):
    """
    Rotate image
    """

    def __init__(self,
                 angle: float = 0.,
                 fill_value: float = 0.0,
                 **kwargs):
        super(Rotate, self).__init__(**kwargs)
        self.angle = angle
        self.fill_value = fill_value

    def _transform_array(self, array):
        return ndimage.rotate(
            array,
            self.angle,
            reshape=False,
            mode='constant',
            cval=self.fill_value)


class Window(ArrayTransform):
    """
    Windows for screening Hounsfield units
    """

    WINDOWS = dict(
        bone=(500, 2000),
        lung=(-100, 2000),
        abdomen=(40, 400),
        brain=(30, 70),
        soft_tissue=(40, 350),
        liver=(40, 300))

    def __init__(self,
                 window_level: int = 0,
                 window_width: int = 2000,
                 window: Union[str, None] = None,
                 apply_to_roi: bool = False,
                 **kwargs):
        super(Window, self).__init__(apply_to_roi=apply_to_roi, **kwargs)
        if window is not None:
            self.window_level, self.window_width = self.WINDOWS[window]
        else:
            self.window_level = window_level
            self.window_width = window_width

    def _transform_array(self, array):
        _lower = self.window_level - self.window_width / 2
        _upper = self.window_level + self.window_width / 2
        array[array <= _lower] = _lower
        array[array >= _upper] = _upper
        return (array - _lower) / self.window_width


class Mask(ArrayTransform):
    """
    Mask values from array
    """

    def __init__(self,
                 lower_value: float = 0.,
                 upper_value: float = 1.,
                 apply_to_roi: bool = False,
                 erosion_disk_size: int = 2,
                 dilation_disk_size: int = 2,
                 mask_value: float = 0.,
                 **kwargs):
        super(Mask, self).__init__(apply_to_roi=apply_to_roi, **kwargs)
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.erosion_disk_size = erosion_disk_size
        self.dilation_disk_size = dilation_disk_size
        self.mask_value = mask_value

    def _transform_array(self, array):
        # build mask: screen all values outside range
        mask = np.where(
            (array < self.lower_value) |
            (array > self.upper_value), 0, 1)
        # find connected regions
        labels = measure.label(mask, background=0.)
        # add the region connected to the uppermost pixel
        # (background, e.g.) to the mask
        background_label = labels[0, 0]
        mask[background_label == labels] = 0
        # slightly smooth the mask
        erosion_mask = morphology.erosion(
            np.where(mask == 1, 1, 0),
            morphology.disk(self.erosion_disk_size))
        dilated_mask = morphology.dilation(
            erosion_mask,
            morphology.disk(self.dilation_disk_size))
        return np.where(dilated_mask == 1, array, self.mask_value)
