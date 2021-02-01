import unittest

from dicomml.cases.case import DicommlCase
from dicomml.transforms import array as transforms
from dicomml.transforms import expand

from tests import sample_case_config


class TestArrayTransforms(unittest.TestCase):

    def test_shift(self):
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Shift(
            x_shift=2, y_shift=34)([case])[0]
        self.assertEqual(
            case.export()['images'].shape,
            case_new.export()['images'].shape)

    def test_mirror(self):
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Mirror()([case])[0]
        self.assertEqual(
            case.export()['images'].shape,
            case_new.export()['images'].shape)

    def test_cut(self):
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Cut(
            x_range=[30, 90],
            y_range=[20, 100]
        )([case])[0]
        self.assertEqual(
            (1, case.export()['images'].shape[1], 60, 80),
            case_new.export()['images'].shape)

    def test_pad(self):
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Pad(target_shape=[200, 150])([case])[0]
        self.assertEqual(
            (1, case.export()['images'].shape[1], 200, 150),
            case_new.export()['images'].shape)
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Pad(target_shape=[120, 120])([case])[0]
        self.assertEqual(
            (1, case.export()['images'].shape[1], 120, 120),
            case_new.export()['images'].shape)

    def test_rotate(self):
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Rotate(angle=10.)([case])[0]
        self.assertEqual(
            case.export()['images'].shape,
            case_new.export()['images'].shape)

    def test_window(self):
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Window(window='lung')([case])[0]
        self.assertEqual(
            case.export()['images'].shape,
            case_new.export()['images'].shape)

    def test_mask(self):
        case = DicommlCase(**sample_case_config())
        case_new = transforms.Mask()([case])[0]
        self.assertEqual(
            case.export()['images'].shape,
            case_new.export()['images'].shape)


class TestExpandTransforms(unittest.TestCase):

    def test_split(self):
        case = DicommlCase(**sample_case_config(n_images=10))
        cases_new = expand.Split(n_images=2)([case])
        self.assertEqual(len(cases_new), 5)
        for case in cases_new:
            self.assertIsInstance(case, DicommlCase)

    def test_add_transforms(self):
        case = DicommlCase(**sample_case_config(n_images=10))
        cases_new = expand.AddTranforms(
            base_transform=transforms.Rotate,
            value_ranges=dict(angle=(-10., 10.)),
            base_config=dict(fill_value=-100.),
            n_applications=6)([case])
        self.assertEqual(len(cases_new), 7)
        for case in cases_new:
            self.assertIsInstance(case, DicommlCase)
