import numpy as np
from skimage.morphology import erosion, dilation
from skimage import measure
from skimage.morphology import disk
#
from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class MaskImages(DicommlTransform):
    """
    Mask images to highlight the relevant features
    """

    def __init__(self,
                 image_lower_HU=-1024,
                 image_upper_HU=-150,
                 image_upper_normalize_HU=-1024,
                 image_lower_normalize_HU=400,
                 erosion_disk_size=2,
                 dilation_disk_size=2,
                 **kwargs):
        super(MaskImages, self).__init__(**kwargs)
        self.image_lower_HU = image_lower_HU
        self.image_upper_HU = image_upper_HU
        self.image_upper_normalize_HU = image_upper_normalize_HU
        self.image_lower_normalize_HU = image_lower_normalize_HU
        self.erosion_disk_size = erosion_disk_size
        self.dilation_disk_size = dilation_disk_size

    def transform_case(self, case: DicommlCase) -> DicommlCase:
        return DicommlCase(
            caseid=case.caseid,
            images={
                key: self._mask_image(arr)
                for key, arr in case.images.items()},
            images_metadata=case.images_metadata,
            rois=case.rois,
            diagnose=case.diagnose,
            images_to_diagnosis=case.images_to_diagnosis,
            images_to_rois=case.images_to_rois)

    def _mask_image(self, _arr):
        """
        Mask image
        """
        mask = np.where(
            (_arr < self.image_lower_HU) |
            (_arr > self.image_upper_HU), 2, 1)
        labels = measure.label(mask)
        background_label = labels[0, 0]
        mask[background_label == labels] = 2
        erosion_mask = erosion(
            np.where(mask == 1, 1, 0),
            disk(self.erosion_disk_size))
        dilated_mask = dilation(
            erosion_mask,
            disk(self.dilation_disk_size))
        # normalize array
        _arr = (_arr - self.image_lower_normalize_HU) \
            / (self.image_upper_normalize_HU - self.image_lower_normalize_HU)
        _arr[_arr > 1] = 1.
        _arr[_arr < 0] = 0.
        return np.where(dilated_mask == 1, _arr, 0)
