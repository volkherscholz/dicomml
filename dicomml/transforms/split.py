from typing import List
#
from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class SplitCase(DicommlTransform):

    def __init__(self,
                 n_images: int = 10,
                 drop_remainder: bool = False,
                 order_images_with_index: bool = True,
                 **kwargs):
        super(SplitCase, self).__init__(**kwargs)
        self.n_images = n_images
        self.drop_remainder = drop_remainder
        self.order_images_with_index = order_images_with_index

    def transform_case(self, case: DicommlCase) -> List[DicommlCase]:
        """
        Split the current case into multiple cases,
        each having n_images of images
        """
        if self.order_images_with_index:
            image_keys = sorted(self.images.keys())
        else:
            image_keys = self.images.keys()
        image_groups = [
            list(image_keys)[i:i + self.n_images]
            for i in range(0, len(image_keys), self.n_images)]
        if self.drop_remainder:
            if len(image_groups[-1]) != self.n_images:
                image_groups = image_groups[:-1]
        return [DicommlCase(
            images={img: case.images[img] for img in group},
            images_metadata={img: case.images_metadata[img]
                for img in group if img in case.images_metadata.keys()},
            rois={
                roikey: roi for roikey, roi in case.rois.items()
                if roikey in [
                    _key for img in group
                    for _key in case.images_to_rois[img]]},
            diagnose={
                diagkey: diag for diagkey, diag in case.diagnose.items()
                if diagkey in [
                    _key for img in group
                    for _key in case.images_to_diagnosis[img]]},
            images_to_diagnosis={
                img: case.images_to_diagnosis[img]
                for img in group if img in case.images_to_diagnosis.keys()},
            images_to_rois={
                img: case.images_to_rois[img]
                for img in group if img in case.images_to_rois.keys()}
            ) for group in image_groups]
