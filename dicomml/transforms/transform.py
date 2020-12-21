from typing import List, Union
from abc import ABC, abstractmethod
import logging
#
from dicomml.cases.case import DicommlCase


class DicommlTransform(ABC):
    """
    Base class of all transformations. A transformation is
    an operation which takes a list of cases and returns a
    possible longer list of cases.
    """

    def __init__(self, name: Union[str, None] = None):
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)

    def transform(self, cases: List[DicommlCase]) -> List[DicommlCase]:
        transformed_cases = []
        for case in cases:
            output = self.transform_case(case)
            if isinstance(output, list):
                transformed_cases = transformed_cases + output
            else:
                transformed_cases.append(output)
        return transformed_cases

    @abstractmethod
    def transform_case(self,
                       case: DicommlCase) -> Union[
                           DicommlCase, List[DicommlCase]]:
        pass

    def __call__(self, cases: List[DicommlCase]) -> List[DicommlCase]:
        return self.transform(cases)
