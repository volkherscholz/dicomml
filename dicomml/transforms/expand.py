import numpy as np
from typing import List, Type, Dict
#
from dicomml.transforms import DicommlTransform, ArrayTransform
from dicomml.cases.case import DicommlCase


class Split(DicommlTransform):

    def __init__(self,
                 n_images: int = 10,
                 drop_remainder: bool = False,
                 order_images_with_index: bool = True,
                 **kwargs):
        super(Split, self).__init__(**kwargs)
        self.n_images = n_images
        self.drop_remainder = drop_remainder
        self.order_images_with_index = order_images_with_index

    def transform_case(self, case: DicommlCase) -> List[DicommlCase]:
        """
        Split the current case into multiple cases,
        each having n_images of images
        """
        if self.order_images_with_index:
            image_keys = sorted(case.images.keys())
        else:
            image_keys = case.images.keys()
        image_groups = [
            list(image_keys)[i:i + self.n_images]
            for i in range(0, len(image_keys), self.n_images)]
        if self.drop_remainder:
            if len(image_groups[-1]) != self.n_images:
                image_groups = image_groups[:-1]
        # collect images etc by group
        case_configs = []
        for group in image_groups:
            images = {}
            images_metadata = {}
            rois = {}
            diagnose = {}
            images_to_diagnosis = {}
            images_to_rois = {}
            for img in group:
                images.update({img: case.images[img]})
                if img in case.images_metadata.keys():
                    images_metadata.update({img: case.images_metadata[img]})
                if img in case.images_to_diagnosis.keys():
                    label_keys_for_image = case.images_to_diagnosis[img]
                    images_to_diagnosis.update({img: label_keys_for_image})
                    diagnose.update({
                        key: val
                        for key, val in case.diagnose.items()
                        if key in label_keys_for_image})
                if img in case.images_to_rois.keys():
                    roi_keys_for_image = case.images_to_rois[img]
                    images_to_rois.update({img: roi_keys_for_image})
                    rois.update({
                        key: val
                        for key, val in case.rois.items()
                        if key in roi_keys_for_image})
            case_configs.append(dict(
                images=images,
                images_metadata=images_metadata,
                rois=rois,
                diagnose=diagnose,
                images_to_diagnosis=images_to_diagnosis,
                images_to_rois=images_to_rois))
        return [DicommlCase(**cfg) for cfg in case_configs]


class AddTranforms(DicommlTransform):

    def __init__(self,
                 base_transform: Type[ArrayTransform],
                 base_config: dict = dict(),
                 value_ranges: Dict[str, tuple] = dict(),
                 n_applications: int = 1,
                 **kwargs):
        transform_configs = []
        for _ in range(n_applications):
            random_config = {
                key: np.random.uniform(low=val[0], high=val[1])
                for key, val in value_ranges.items()}
            transform_configs.append({
                **base_config,
                **random_config})
        self.random_instances = [
            base_transform(**cfg) for cfg in transform_configs]

    def transform_case(self, case: DicommlCase) -> List[DicommlCase]:
        """
        Expand images by adding transformed versions
        """
        cases = [case]
        for transform in self.random_instances:
            cases.append(transform([case])[0])
        return cases
