from typing import List, Union
from abc import ABCMeta, abstractmethod
#
from dicomml.cases.case import DicommlCase


class DicommlTransform(ABCMeta):
    """
    Base class of all transformations. A transformation is
    an operation which takes a list of cases and returns a
    possible longer list of cases.
    """

    def __init__(self, name):
        self.name = name

    def transform(self, cases: List[DicommlCase]) -> List[DicommlCase]:
        transformed_cases = []
        for case in cases:
            output = self.transform_case(case)
            if type(output) == list:
                transformed_cases = transformed_cases + output
            else:
                transformed_cases.append(output)
        return transformed_cases

    @abstractmethod
    def transform_case(self,
                       case: DicommlCase) -> Union[
                           DicommlCase, List[DicommlCase]]:
        pass
