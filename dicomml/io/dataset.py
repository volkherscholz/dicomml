from typing import Dict, Union
import glob
import os
#
import tensorflow as tf
#
from dicomml.cases.case import DicommlCase


class DicommlDataset(tf.data.Dataset):
    """
    An extension of the usual tensorflow dataset API
    for use with DicommlCases. In particular, the main usage
    is as follows: given directory with DicommlCases saved as
    zip files, split the set of files into training, evaluation
    and test dataset and construct a dataset with each element
    containing images as well as as rois and or labels if applicable
    """

    @classmethod
    def from_folder(cls,
                    path: str = '.',
                    split_dataset: bool = False,
                    split_ratios: Dict[str, float] = dict(
                        training=0.8,
                        evaluation=0.1,
                        test=0.1),
                    **kwargs
                    ) -> Union['DicommlDataset', Dict[str, 'DicommlDataset']]:
        files = [
            _f for _f in glob.glob(path, recursive=True)
            if os.path.isfile(_f)]
        if not split_dataset:
            return super(DicommlDataset).from_tensor_slices(files).map(
                lambda file: DicommlCase.load(file).export(**kwargs))
        if sum(split_ratios) != 1:
            raise ValueError('Split ratios have to add up to one.')
        datasets = dict()
        _prev_ratio = 0
        for name, ratio in split_ratios.items():
            _lower_index = int(_prev_ratio * len(files))
            _upper_index = int((_prev_ratio + ratio) * len(files))
            datasets.update({name: files[_lower_index:_upper_index]})
            _prev_ratio = ratio
        # add remaining files to last dataset
        if _upper_index < len(files):
            datasets[list(datasets.keys())[-1]].append(files[_upper_index:])
        # construct datasets
        return {
            name: super(DicommlDataset).from_tensor_slices(_files).map(
                lambda _file: DicommlCase.load(_file).export(**kwargs))
            for name, _files in datasets}
