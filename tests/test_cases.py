import unittest
import tempfile

from dicomml.cases.case import DicommlCase

from tests import sample_case_config


class TestDicommlCase(unittest.TestCase):

    def test_create(self):
        case = DicommlCase(**sample_case_config())
        self.assertIsInstance(case, DicommlCase)

    def test_save_load(self):
        case = DicommlCase(**sample_case_config())
        with tempfile.TemporaryDirectory() as temp_folder:
            zipfile = case.save(path=temp_folder)
            case_loaded = DicommlCase.load(zipfile)
        self.assertEqual(case, case_loaded)

    def test_export(self):
        case = DicommlCase(**sample_case_config())
        exports = case.export()
        self.assertCountEqual(
            list(exports.keys()),
            ['images', 'truth'])
        self.assertEqual(exports['images'].shape, (10, 120, 120, 1))
