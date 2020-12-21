import os
import glob
import uuid
from tempfile import mkdtemp
from shutil import make_archive, unpack_archive, rmtree
import numpy as np
import json
import pandas as pd

from typing import Union, List, Iterator, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
                 caseid: Union[str, None] = None,
                 images: dict = {},
                 images_metadata: dict = {},
                 rois: dict = {},
                 diagnose: dict = {},
                 images_to_diagnosis: dict = {},
                 images_to_rois: dict = {}):
        if caseid is None:
            caseid = str(uuid.uuid4())
        self.caseid = caseid
        # dictionary of images
        self.images = images
        # image metadata
        self.images_metadata = images_metadata
        # dictionary of rois
        self.rois = rois
        # dictionary of diagnoses
        self.diagnose = diagnose
        # dictionary linking images & diagnosis
        self.images_to_diagnosis = images_to_diagnosis
        # dictionary linking images to rois
        self.images_to_rois = images_to_rois

    def add_rois(self,
                 rois: pd.DataFrame,
                 rois_index_column: List[str] = ['z_coordinate', 'RoiNo'],
                 x_coord_column: str = 'x_coordinate',
                 y_coord_column: str = 'y_coordinate'):
        """
        Add roi information
        Args:
            rois: pandas Dataframe with at least three columns
                one specifying the image identifier (default instanceNr)
                one specifying the column with x coordinates
                one specifying the column with y coordinates
        """
        rois = rois.set_index(rois_index_column)
        rois = rois[[x_coord_column, y_coord_column]].to_dict(orient='index')

        def construct_roi(image_shape, x_coord, y_coord):
            import scipy.interpolate as si
            z_values = np.ones(np.array(x_coord).shape)
            points = np.array([x_coord, y_coord]).transpose()
            grid = np.array(np.meshgrid(
                np.arange(0, image_shape[1]),
                np.arange(0, image_shape[0]))).transpose()
            interp = si.griddata(
                points, z_values, grid, method='cubic', fill_value=0.)
            return interp.transpose()

        for (imgkey, _), roidata in rois.items():
            roikey = str(uuid.uuid4())
            if isinstance(imgkey, float):
                imgkey = round(imgkey, 2)
            self.rois.update({
                roikey: construct_roi(
                    image_shape=self.images[imgkey].shape,
                    x_coord=roidata[x_coord_column],
                    y_coord=roidata[y_coord_column])})
            if imgkey in self.images_to_rois.keys():
                self.images_to_rois[imgkey].append(roikey)
            else:
                self.images_to_rois.update({imgkey: [roikey]})

    def add_diagnose(self,
                     diagnoses: pd.DataFrame,
                     diagnose_labels: list,
                     image_index_column: str = 'instanceNr',
                     diagnose_column: str = 'keywords'):
        """
        Add diagnose information
        structure of DataFrame:
        - image_index_column: which image is meant
        - diagnose_column: the diagnose (string)
        - diagnose_labels: list of unique labels
        """
        # get unique diagnose labels
        _diag_labels = sorted(diagnoses[diagnose_column].unique().tolist())
        for diag_label in _diag_labels:
            diagkey = str(uuid.uuid4())
            self.diagnose.update({diagkey: diag_label})
            images = diagnoses[
                diagnoses[diagnose_column] == diag_label][
                    image_index_column].tolist()
            for img in images:
                if img in self.images_to_diagnoses.key():
                    self.images_to_diagnoses[img].append(diagkey)
                else:
                    self.images_to_diagnoses.update({img: [diagkey]})

    def save(self, path: str = '.'):
        """
        Saves the current case as zip file
        """
        tempfolder = mkdtemp()
        _path = os.path.join(tempfolder, self.caseid)
        os.makedirs(_path, exist_ok=True)
        with open(os.path.join(_path, 'meta.json'), 'w') as _f:
            json.dump({
                'images_metadata': self.images_metadata,
                'diagnose': self.diagnose,
                'images_to_diagnosis': self.images_to_diagnosis,
                'images_to_rois': self.images_to_rois
            }, _f)
        if len(self.images) > 0:
            with open(os.path.join(_path, 'images.npz'), 'wb') as _f:
                np.savez_compressed(
                    _f,
                    **{str(key): arr for key, arr in self.images.items()})
        if len(self.rois) > 0:
            with open(os.path.join(_path, 'rois.npz'), 'wb') as _f:
                np.savez_compressed(_f, **self.rois)
        file_path = os.path.join(path, self.caseid)
        file_path = make_archive(
            base_name=file_path,
            format='zip',
            root_dir=tempfolder,
            base_dir=self.caseid)
        return file_path

    @classmethod
    def load(cls, zipfile):
        """
        Load a case from a zipfile
        """
        tempfolder = mkdtemp()
        unpack_archive(zipfile, tempfolder)
        # assumes that the zipfile has a basepath
        caseid = os.listdir(tempfolder)[0]
        _path = os.path.join(tempfolder, caseid)
        # load metadata
        with open(os.path.join(_path, 'meta.json'), 'r') as _f:
            _meta = json.load(_f)
        images_metadata = _meta['images_metadata']
        diagnose = _meta['diagnose']
        images_to_diagnosis = _meta['images_to_diagnosis']
        images_to_rois = _meta['images_to_rois']
        # load images
        if os.path.isfile(os.path.join(_path, 'images.npz')):
            with open(os.path.join(_path, 'images.npz'), 'rb') as _f:
                images = np.load(_f)
                # try to convert keys to floats
                # for the z coordinate
                try:
                    images = {float(key): arr for key, arr in images.items()}
                except ValueError:
                    pass
        else:
            images = {}
        if os.path.isfile(os.path.join(_path, 'rois.npz')):
            with open(os.path.join(_path, 'images.npz'), 'rb') as _f:
                rois = np.load(_f)
        else:
            rois = {}
        return cls(
            caseid=caseid,
            images=images,
            rois=rois,
            images_metadata=images_metadata,
            diagnose=diagnose,
            images_to_diagnosis=images_to_diagnosis,
            images_to_rois=images_to_rois)

    def export(self,
               include_diagnoses: bool = True,
               diagnose_label_set: List[str] = [],
               include_rois: bool = True,
               order_images_with_index: bool = True):
        """
        Exports the case for training or inference as dictionary
        holding arrays with images, rois and one-hot encoded labels
        Args:
            include_diagnose: whether to include the diagnoses as
                one-hot encoded vector
            diagnose_label_set: the base set of diagnose labels
                which is used to create a one-hot encoded vector
                i.e. the length of the vector is the length of this set
                and for each image we check whether the label is present
                or not
            include_rois: whether to include the rois
            order_images_with_index: whether to order the images according to
                their key, i.e. instanceNr before exporting
        """
        if order_images_with_index:
            image_keys = sorted(self.images.keys())
        else:
            image_keys = self.images.keys()
        # construct arrays
        _image_array = []
        _roi_array = []
        _diagnose_array = []
        # iterate
        for imgkey in image_keys:
            _image_array.append(self.images[imgkey])
            if imgkey in self.images_to_rois.keys():
                _rois = np.array(sum([
                    self.rois[_key] for _key in self.images_to_rois[imgkey]]))
                # restict to maximal value of 1 if multiple rois overlay
                _roi_array.append(np.where(_rois > 1., 1., _rois))
            else:
                # zero rois
                _roi_array.append(np.zeros(self.images[imgkey].shape))
            if imgkey in self.images_to_diagnosis.keys():
                _diagnose_labels_img = [
                    self.diagnose[_key]
                    for _key in self.images_to_diagnosis[imgkey]]
                _diagnose_array.append([
                    1. if diagnose in _diagnose_labels_img else 0.
                    for diagnose in diagnose_label_set])
            else:
                # zero diagnose
                _diagnose_array.append(np.zeros(len(diagnose_label_set)))
        data = dict(images=np.array(_image_array))
        if include_diagnoses:
            data.update(dict(labels=np.array(_diagnose_array)))
        if include_rois:
            data.update(dict(rois=np.array(_roi_array)))
        return data

    def iterate(self) -> Iterator[Tuple[str, np.ndarray, str]]:
        """
        Return images overlayed with rois (if present)
        and labels (if present)
        """
        image_keys = iter(sorted(self.images.keys()))
        while True:
            try:
                image_key = next(image_keys)
                image = self.images[image_key]
                if image_key in self.images_to_rois.keys():
                    image = image + np.array(sum([
                        self.rois[_key]
                        for _key in self.images_to_rois[image_key]]))
                if image_key in self.images_to_diagnosis.keys():
                    labels = 'Labels: {}'.format(' '.join([
                        self.diagnose[_key]
                        for _key in self.images_to_diagnosis[image_key]]))
                else:
                    labels = 'No labels'
                yield str(image_key), image, labels
            except StopIteration:
                return

    def visualize(self, fig, ax) -> Figure:
        contents = list(self.iterate())

        from ipywidgets import interact
        from IPython.display import display

        def slider(slice):
            ax.imshow(contents[slice][1], cmap=plt.cm.gray)
            fig.canvas.draw()
            display(fig)

        return interact(slider, slice=(0, len(contents) - 1))

    @classmethod
    def from_dicom_folder(cls,
                          path,
                          pattern='**',
                          caseid=None,
                          caseid_from_folder_name=True,
                          exlude_dicom_dir=True,
                          scale_to_hounsfield=True,
                          index_image='z_coordinate',
                          delete_folder: bool = False):
        """
        Create a case from a folder of dicom images
        """
        files = [
            _f for _f in
            glob.glob(os.path.join(path, pattern), recursive=True)
            if os.path.isfile(_f)]
        if exlude_dicom_dir:
            files = [_f for _f in files if 'DICOMDIR' not in _f]
        images = {}
        images_metadata = {}
        for _f in files:
            metadata, _arr = cls._read_dicom(_f, scale_to_hounsfield)
            index = metadata.pop(index_image)
            images.update({index: _arr})
            images_metadata.update({index: metadata})
        if caseid_from_folder_name:
            caseid = os.path.basename(path)
        if delete_folder:
            rmtree(path)
        return cls(
            caseid=caseid,
            images=images,
            images_metadata=images_metadata)

    @classmethod
    def from_dicom_zipfile(cls, zipfile: str, **kwargs):
        """
        Create a case from a zipfile containing a folder with
        dicom images
        """
        tempfolder = mkdtemp()
        unpack_archive(zipfile, tempfolder, format='zip')
        return cls.from_dicom_folder(
            tempfolder, delete_folder=True, **kwargs)

    @staticmethod
    def _read_dicom(filename, scale_to_hounsfield: bool = True):
        """
        Read a dicom file and return
        an array as well as metadata
        - scale_to_hounsfield: whether to scale the data
          to hounsfield units
        """
        _d = dcmread(filename)
        # extract metadata information
        data = {
            'patientid': str(_d[('0010', '0020')].value),
            'studyuid':  str(_d[('0020', '000d')].value),
            'seriesuid': str(_d[('0020', '000e')].value),
            'instanceNr': int(_d[('0020', '0013')].value),
            'seriesNr': str(_d[('0020', '0011')].value),
            'sopuid': str(_d[('0008', '0018')].value),
            'x_coordinate': round(float(_d[('0020', '0032')].value[0]), 2),
            'y_coordinate': round(float(_d[('0020', '0032')].value[1]), 2),
            'z_coordinate': round(float(_d[('0020', '0032')].value[2]), 2)
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
