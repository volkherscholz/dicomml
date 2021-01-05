import os
from typing import Union, List, Dict

from dicomml.transforms import DicommlTransform
from dicomml.cases.case import DicommlCase


class Load(DicommlTransform):
    """
    Load DicommlCases from folder
    """

    def __init__(self,
                 load_from_dicom: bool = False,
                 load_config: dict = dict(),
                 **kwargs):
        super(Load, self).__init__(**kwargs)
        self.load_from_dicom = load_from_dicom
        self.load_config = load_config

    def __call__(self, files: List[str] = []) -> List[DicommlCase]:
        if self.load_from_dicom:
            cases = [
                DicommlCase.from_dicom_zipfile(_file, **self.load_config)
                for _file in files]
        else:
            cases = [
                DicommlCase.load(_file)
                for _file in files]
            self.logger.info('Loaded cases: {}'.format(
                [case.caseid for case in cases]))
        return cases


class Save(DicommlTransform):
    """
    Load DicommlCases from folder
    """

    def __init__(self,
                 split_ratios: Union[Dict[str, float], None] = None,
                 shuffle_cases: bool = False,
                 **kwargs):
        super(Save, self).__init__(**kwargs)
        self.split_ratios = split_ratios
        self.shuffle_cases = shuffle_cases

    def __call__(self,
                 cases: List[DicommlCase],
                 folder: str = '.') -> Dict[str, str]:
        if self.split_ratios is None:
            files = {
                folder: [case.save(folder) for case in cases]}
        else:
            files = {}
            if self.shuffle_cases:
                import random
                random.shuffle(cases)
            for name, _cases in self.split_case_list(cases).items():
                folder_name = os.path.join(folder, name)
                for case in _cases:
                    case.save(folder_name)
                files.update({name: folder_name})
                self.logger.info('Saved cases {} to folder {}'.format(
                    [case.caseid for case in _cases], folder_name))
        return files

    def split_case_list(self, cases):
        if sum(self.split_ratios.values()) != 1:
            raise ValueError('Split ratios have to add up to one.')
        casesets = dict()
        _prev_ratio = 0
        for name, ratio in self.split_ratios.items():
            _lower_index = int(_prev_ratio * len(cases))
            _upper_index = int((_prev_ratio + ratio) * len(cases))
            casesets.update({name: cases[_lower_index:_upper_index]})
            _prev_ratio += ratio
        # add remaining cases to last dataset
        if _upper_index < len(cases):
            casesets[list(casesets.keys())[-1]] += cases[_upper_index:]
        return casesets
