import unittest
import tempfile
import shutil
import glob
import os

from dicomml.cases.case import DicommlCase
from dicomml.tasks.main import run_task, run_pipeline

from tests import sample_case_config


class TestLTS(unittest.TestCase):

    def get_config(self):
        return dict(
            folder_in=self.folder_in,
            num_concurrent_files=1,
            steps={
                'transforms.expand.Split': dict(
                    n_images=5),
                'transforms.expand.AddTranforms': dict(
                    base_transform='transforms.array.Rotate',
                    n_applications=10,
                    value_ranges=dict(angle=(-20, 20)))},
            folder_out=self.folder_out,
            save_config=dict(
                shuffle_cases=True,
                split_ratios=dict(
                    train=0.8,
                    eval=0.1,
                    test=0.1)))

    def setUp(self):
        self.folder_in = tempfile.mkdtemp()
        self.folder_out = tempfile.mkdtemp()
        for i in range(4):
            DicommlCase(**sample_case_config(
                caseid='case-{i}'.format(i=i),
                n_images=10
            )).save(self.folder_in)

    def test_sequential_task(self):
        case_sets = run_task(
            task_class='tasks.tasks.DicommlLTS',
            config=self.get_config())
        self.assertCountEqual(
            list(case_sets.keys()),
            ['train', 'eval', 'test'])
        case_files = glob.glob(os.path.join(self.folder_out, '*', '*.zip'))
        self.assertEqual(len(case_files), 88)
        shutil.rmtree(self.folder_out)

    def test_parallel_task(self):
        import glob
        files = glob.glob(os.path.join(self.folder_in, '*.zip'))
        parallel = [
            dict(filenames=files[0:2]),
            dict(filenames=files[2:4])]
        run_pipeline(tasks=dict(lts=dict(
                task_class='tasks.tasks.DicommlLTS',
                config=self.get_config(),
                parallel_configs=parallel)))
        case_files = glob.glob(os.path.join(self.folder_out, '*', '*.zip'))
        self.assertEqual(len(case_files), 88)
        shutil.rmtree(self.folder_out)

    def tearDown(self):
        shutil.rmtree(self.folder_in)
