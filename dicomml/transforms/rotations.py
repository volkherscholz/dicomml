import numpy as np
from scipy import ndimage
from typing import List
#
from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class AddRotatedImages(DicommlTransform):

    def __init__(self,
                 lower_angle=-45,
                 upper_angle=45,
                 n_rotations=1,
                 **kwargs):
        super(AddRotatedImages, self).__init__(**kwargs)
        self.lower_angle = lower_angle
        self.upper_angle = upper_angle
        self.n_rotations = n_rotations

    def transform_case(self, case: DicommlCase) -> List[DicommlCase]:
        """
        Expand images by adding transformed versions
        """
        rotated_images = {}
        rotated_rois = {}
        cases = [case]
        #
        for i in range(self.n_rotations):
            for index, arr in case.images:
                angle = np.random.uniform(
                    low=self.lower_angle, high=self.upper_angle)
                rotated_images.update({
                    index: ndimage.rotate(arr, angle, reshape=False)
                })
                if index in case.images_to_rois.keys():
                    roi_index = case.images_to_rois[index]
                    roi = case.rois[roi_index]
                    rotated_rois.update({
                        roi_index: ndimage.rotate(roi, angle, reshape=False)
                    })
            cases.append(DicommlCase(
                caseid=case.caseid + str(i + 1),
                images=rotated_images,
                rois=rotated_rois,
                images_metadata=case.images_metadata,
                diagnose=case.diagnose,
                images_to_diagnosis=case.images_to_diagnosis,
                images_to_rois=case.images_to_rois
            ))
        return cases
