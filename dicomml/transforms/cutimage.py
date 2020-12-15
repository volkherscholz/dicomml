from typing import List, Union
#
from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class CutImages(DicommlTransform):
    """
    Mask images to highlight the relevant features
    """

    def __init__(self,
                 x_range: List[Union[int, None]] = [None, None],
                 y_range: List[Union[int, None]] = [None, None],
                 **kwargs):
        super(CutImages, self).__init__(**kwargs)
        self.x_range = x_range
        self.y_range = y_range

    def transform_case(self, case: DicommlCase) -> DicommlCase:
        return DicommlCase(
            caseid=case.caseid,
            images={
                key: self._cut_image(arr)
                for key, arr in case.images.items()},
            images_metadata=case.images_metadata,
            rois=case.rois,
            diagnose=case.diagnose,
            images_to_diagnosis=case.images_to_diagnosis,
            images_to_rois=case.images_to_rois)

    def _cut_image(self, array):
        return array[
            self.x_range[0]:self.x_range[1],
            self.y_range[0]:self.y_range[1]]
