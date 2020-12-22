import numpy as np
from scipy import ndimage
from typing import List
#
from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class AddRotatedImages(DicommlTransform):

    def __init__(self,
                 lower_angle: float = -45.0,
                 upper_angle: float = 45.0,
                 n_rotations: int = 1,
                 fill_value: float = 0.,
                 **kwargs):
        super(AddRotatedImages, self).__init__(**kwargs)
        self.lower_angle = lower_angle
        self.upper_angle = upper_angle
        self.n_rotations = n_rotations
        self.fill_value = fill_value

    def transform_case(self, case: DicommlCase) -> List[DicommlCase]:
        """
        Expand images by adding transformed versions
        """
        cases = [case]
        #
        for i in range(self.n_rotations):
            cases.append(DicommlCase(
                caseid='{}-{}'.format(case.caseid, str(i + 1)),
                images={
                    key: self._rotate_image(arr)
                    for key, arr in case.images.items()},
                images_metadata=case.images_metadata,
                rois={
                    key: self._rotate_image(arr)
                    for key, arr in case.rois.items()},
                diagnose=case.diagnose,
                images_to_diagnosis=case.images_to_diagnosis,
                images_to_rois=case.images_to_rois))
        return cases

    def _rotate_image(self, array):
        angle = np.random.uniform(low=self.lower_angle, high=self.upper_angle)
        return ndimage.rotate(
            array, angle, reshape=False, mode='constant', cval=self.fill_value)
