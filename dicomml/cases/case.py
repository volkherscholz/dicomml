import os
import glob
import uuid
import numpy as np
#
from pydicom.filereader import dcmread


class DicommlCase:
    """
    Base class of a Dicomml Case. A Case is a collection
    of images, diagnoses as well as regions of interests.
    Images are numpy tensors, diagnoses are strings, and
    regions of interests are numpy tensors with 0/1 entries
    masking the regions of interest in an image.
    In addition, a case holds dictionaries which link the
    three. The usual creation process of a case is from a
    folder of dicom images, but more implementations exists.
    The base implementation provides methods to split a case
    into more, expand the image data by e.g. rotation and
    add diagnoses as well as regions of interest as well as
    some i/o functions.
    """

    def __init__(self,
                 caseid=None,
                 images: dict = {},
                 rois: dict = {},
                 diagnose: dict = {},
                 images_to_diagnosis: dict = {},
                 images_to_rois: dict = {}):
        if caseid is None:
            caseid = uuid.uuid4()
        self.caseid = caseid
        # dictionary of images
        self.images = images
        # dictionary of rois
        self.rois = rois
        # dictionary of diagnoses
        self.diagnose = diagnose
        # dictionary linking images & diagnosis
        self.images_to_diagnosis = images_to_diagnosis
        # dictionary linking images to rois
        self.images_to_rois = images_to_rois

    def expand_images(self, method='rotation'):
        """
        expand images by adding transformed versions
        Current implementation supports rotation
        """

    def split(self, nimages=10):
        """
        Split the current case into multiple cases,
        each having nimages of images
        """

    def save(self, path):
        """
        Saves the current case as zip file
        """
        file_path = os.path.join(path, self.caseid + '.zip')
        return file_path

    def export(self,
               includediagnoses=True,
               includerois=True,
               format='tfexample'):
        """
        Exports the case. Currently the tfexample
        format is supported.
        """

    @classmethod
    def from_dicom_folder(cls,
                          path,
                          pattern='*',
                          exlude_dicom_dir=True,
                          scale_to_hounsfield=True,
                          index_image='instanceNr'):
        """
        Create a case from a folder of dicom images
        """
        files = glob.glob(os.path.join(path, pattern))
        if exlude_dicom_dir:
            files = [_f for _f in files if _f != 'DICOMDIR']
        images = {}
        for _f in files:
            metadata, _arr = cls._read_dicom(_f, scale_to_hounsfield)
            index = metadata.pop(index_image)
            images.update({
                index: {
                    'metadata': metadata,
                    'image': _arr
                }})
        return cls(images=images)

    @staticmethod
    def _read_dicom(self, filename, scale_to_hounsfield):
        """
        Read a dicom file and return
        an array as well as metadata
        - scale_to_hounsfield: whether to scale the data
          to hounsfield units
        """
        _d = dcmread(filename)
        # extract metadata information
        data = {
            'patientid': _d[('0010', '0020')].value,
            'studyuid':  _d[('0020', '000d')].value,
            'seriesuid': _d[('0020', '000e')].value,
            'instanceNr': _d[('0020', '0013')].value,
            'seriesNr': _d[('0020', '0011')].value,
            'sopuid': _d[('0008', '0018')].value,
            'position': _d[('0020', '0032')].value
        }
        _arr = _d.pixel_array.astype(np.uint16)
        if scale_to_hounsfield:
            # remove very negative values
            _arr[_arr == -2000] = 0
            # Convert to Hounsfield units (HU)
            intercept = float(_d.RescaleIntercept)
            slope = float(_d.RescaleSlope)
            if slope != 1:
                _arr = slope * _arr.astype(np.float64)
                _arr = _arr.astype(np.int16)
            _arr = _arr + np.int16(intercept)
        return data, _arr
