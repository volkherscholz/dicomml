from typing import Union, List, Tuple
import numpy as np

import tensorflow as tf

from dicomml.transforms.transform import DicommlTransform
from dicomml.cases.case import DicommlCase


class AddPredictions(DicommlTransform):
    """
    Adds model predictions to a case
    """

    def __init__(self,
                 model: tf.keras.Model,
                 model_predicts_labels: bool = False,
                 model_predicts_rois: bool = False,
                 order_images_with_index: bool = True,
                 diagnose_label_set: List[str] = [],
                 label_threshold: float = 0.5,
                 **kwargs):
        super(AddPredictions, self).__init__(**kwargs)
        self.model = model
        self.model_predicts_labels = model_predicts_labels
        self.model_predicts_rois = model_predicts_rois
        self.order_images_with_index = order_images_with_index
        self.diagnose_label_set = diagnose_label_set
        self.label_threshold = label_threshold

    def transform_case(self, case: DicommlCase) -> DicommlCase:
        images = case.export(
            order_images_with_index=self.order_images_with_index
            )['images']
        labels, rois = self._model_predictions(images)
        if self.order_images_with_index:
            image_keys = sorted(case.images.keys())
        else:
            image_keys = case.images.keys()
        labels_names = {
            str(i): diagnose
            for i, diagnose in enumerate(self.diagnose_label_set)}
        case_images_to_labels = {}
        case_rois = {}
        case_images_to_rois = {}
        for i, imgkey in enumerate(image_keys):
            if rois is not None:
                case_rois.update({str(i): rois[i, ...]})
                case_images_to_rois.update({imgkey: [str(i)]})
            if labels is not None:
                _logits = labels[i, ...].tolist()
                _labels = [
                    j for j, _ in enumerate(self.diagnose_label_set)
                    if _logits[j] >= self.label_threshold]
                case_images_to_labels.update({imgkey: _labels})
        return DicommlCase(
            caseid=case.caseid,
            images=case.images,
            images_metadata=case.images_metadata,
            rois=rois,
            diagnose=labels_names,
            images_to_diagnosis=case_images_to_labels,
            images_to_rois=case_images_to_rois)

    def _model_predictions(self,
                           images) -> Tuple[
                               Union[np.ndarray, None],
                               Union[np.ndarray, None]]:
        if self.model_predicts_labels and self.model_predicts_rois:
            labels, rois = self.model.predict(images)
        elif self.model_predicts_labels and (not self.model_predicts_rois):
            labels = self.model.predict(images)
            rois = None
        elif (not self.model_predicts_labels) and self.model_predicts_rois:
            rois = self.model.predict(images)
            labels = None
        else:
            self.logger.warn('No model predictions set.')
            labels, rois = None, None
        return labels, rois
