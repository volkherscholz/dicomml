import numpy as np
#
from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class Shift(DicommlTransform):
    """
    Mask images to highlight the relevant features
    """

    def __init__(self,
                 x_shift: int = 0,
                 y_shift: int = 0,
                 shift_images: bool = False,
                 shift_rois: bool = False,
                 **kwargs):
        super(Shift, self).__init__(**kwargs)
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.shift_images = shift_images
        self.shift_rois = shift_rois

    def transform_case(self, case: DicommlCase) -> DicommlCase:
        if self.shift_images:
            images = {
                key: self._shift_array(arr)
                for key, arr in case.images.items()}
        else:
            images = case.images
        if self.shift_rois:
            rois = {
                key: self._shift_array(arr)
                for key, arr in case.rois.items()}
        else:
            rois = case.rois
        return DicommlCase(
            caseid=case.caseid,
            images=images,
            images_metadata=case.images_metadata,
            rois=rois,
            diagnose=case.diagnose,
            images_to_diagnosis=case.images_to_diagnosis,
            images_to_rois=case.images_to_rois)

    def _shift_array(self, array):
        array = np.roll(array, self.x_shift, axis=0)
        array = np.roll(array, self.y_shift, axis=1)
        return array


class Mirror(DicommlTransform):
    """
    Mask images to highlight the relevant features
    """

    def __init__(self,
                 mirror_image: bool = True,
                 mirror_roi: bool = True,
                 axis: int = -1,
                 **kwargs):
        super(Mirror, self).__init__(**kwargs)
        self.mirror_image = mirror_image
        self.mirror_roi = mirror_roi
        self.axis = axis

    def transform_case(self, case: DicommlCase) -> DicommlCase:
        if self.mirror_image:
            images = {
                key: self._mirror_array(arr)
                for key, arr in case.images.items()}
        else:
            images = case.images
        if self.mirror_roi:
            rois = {
                key: self._mirror_array(arr)
                for key, arr in case.rois.items()}
        else:
            rois = case.rois
        return DicommlCase(
            caseid=case.caseid,
            images=images,
            images_metadata=case.images_metadata,
            rois=rois,
            diagnose=case.diagnose,
            images_to_diagnosis=case.images_to_diagnosis,
            images_to_rois=case.images_to_rois)

    def _mirror_array(self, array):
        return np.flip(array, axis=self.axis)
