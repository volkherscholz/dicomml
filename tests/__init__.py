import random
import numpy as np
import uuid
from typing import Tuple, Union


def sample_case_config(caseid: Union[str, None] = None,
                       n_images: int = 10,
                       n_rois: int = 5,
                       n_labels: int = 3,
                       image_size: Tuple[int, int] = (120, 120)):
    if caseid is None:
        caseid = str(uuid.uuid4())
    image_keys = np.random.uniform(-123.4, 210.0, n_images).tolist()
    images = {
        imgkey: np.random.randint(-1000, 2000, image_size)
        for imgkey in image_keys}
    rois = {
        str(i): np.random.randint(2, size=image_size)
        for i in range(n_rois)}
    diagnose = {
        str(i): 'diagnose-{i}'.format(i=i)
        for i in range(n_labels)}
    images_to_rois = {
        imgkey: random.sample(list(rois.keys()), 2)
        for imgkey in random.sample(list(image_keys), 5)}
    images_to_diagnosis = {
        imgkey: random.sample(list(diagnose.keys()), 2)
        for imgkey in random.sample(list(image_keys), 5)}
    return dict(
        caseid=caseid,
        images=images,
        images_metadata={},
        rois=rois,
        diagnose=diagnose,
        images_to_diagnosis=images_to_diagnosis,
        images_to_rois=images_to_rois)
