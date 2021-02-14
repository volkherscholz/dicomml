from typing import List, Union
import logging
#
from dicomml.cases.case import DicommlCase


class DicommlTransform:
    """
    Base class of all transformations. A transformation is
    an operation which takes a list of cases and returns a
    possible longer list of cases.
    """

    def __init__(self, name: Union[str, None] = None):
        if name is None:
            name = type(self).__name__
        self.name = name
        self.logger = logging.getLogger(type(self).__name__)

    def transform(self, cases: List[DicommlCase]) -> List[DicommlCase]:
        transformed_cases = []
        for case in cases:
            output = self.transform_case(case.copy())
            if isinstance(output, list):
                transformed_cases = transformed_cases + output
            else:
                transformed_cases.append(output)
        return transformed_cases

    def transform_case(self,
                       case: DicommlCase) -> Union[
                           DicommlCase, List[DicommlCase]]:
        return case

    def __call__(self, cases: List[DicommlCase]) -> List[DicommlCase]:
        return self.transform(cases)


class ArrayTransform(DicommlTransform):
    """
    Basic image transforms
    """

    def __init__(self,
                 apply_to_image: bool = True,
                 apply_to_roi: bool = True,
                 **kwargs):
        super(ArrayTransform, self).__init__(**kwargs)
        self.apply_to_image = apply_to_image
        self.apply_to_roi = apply_to_roi

    def transform_case(self, case: DicommlCase) -> DicommlCase:
        if self.apply_to_image:
            images = {
                key: self._transform_array(arr)
                for key, arr in case.images.items()}
        else:
            images = case.images
        if self.apply_to_roi:
            rois = {
                key: self._transform_array(arr)
                for key, arr in case.rois.items()}
        else:
            rois = case.rois
        self.logger.debug('Applied transform to case {}'.format(case.caseid))
        return DicommlCase(
            caseid=case.caseid,
            images=images,
            images_metadata=case.images_metadata,
            rois=rois,
            diagnose=case.diagnose,
            images_to_diagnosis=case.images_to_diagnosis,
            images_to_rois=case.images_to_rois)

    def _transform_array(self, array):
        return array
