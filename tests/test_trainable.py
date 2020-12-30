import unittest
import tempfile
import shutil
import numpy as np
import os

from dicomml.cases.case import DicommlCase
from dicomml.tasks.main import run_task

from tests import sample_case_config


class TestLTS(unittest.TestCase):

    def get_config(self):
        from ray import tune

        scheduler = tune.schedulers.PopulationBasedTraining(
            time_attr="training_step_count",
            metric='BinaryAccuracy',
            mode='max',
            perturbation_interval=2,
            hyperparam_mutations={
                "optimizer_learning_rate": lambda: np.random.uniform(0.001, 1),
                "model_dropoutrate": lambda: np.random.uniform(0.05, 0.15)},
            quantile_fraction=0.5,
            resample_probability=1.0,
            log_config=True,
            require_attrs=True)

        return dict(
             metric='BinaryAccuracy',
             mode='max',
             trainable_config=dict(
                 train_iterations_per_step=2,
                 eval_iterations_per_step=1,
                 train_path=os.path.join(self.folder_in, 'train', '*.zip'),
                 eval_path=os.path.join(self.folder_in, 'eval', '*.zip'),
                 train_batch_size=2,
                 eval_batch_size=2,
                 shuffle_buffer_size=2,
                 data_keys=['images', 'truth'],
                 padded_shapes={'images': [10, 120, 120, 1], 'truth': [10, 120, 120, 1]},  # noqa 501
                 transformations={
                     'transforms.array.Window': dict(window='soft_tissue'),
                     'transforms.expand.AddTranforms': dict(
                        base_transform='transforms.array.Rotate',
                        value_ranges=dict(angle=(-10., 10.)),
                        base_config=dict(fill_value=-100.),
                        n_applications=6)},
                 export_config=dict(
                     include_diagnoses=False,
                     include_rois=True),
                 loss_function=('keras.losses.BinaryCrossentropy', dict()),
                 train_metric=('keras.metrics.BinaryAccuracy', dict()),
                 eval_metrics={
                     'keras.metrics.BinaryAccuracy': dict(),
                     'keras.metrics.BinaryCrossentropy': dict()},
                 model_class='models.unet.UNETModel',
                 optimizer_class='keras.optimizers.Adam'),
             scheduler=scheduler,
             stop={"training_iteration": 4},
             num_samples=2,
             reuse_actors=True,
             resources_per_trial=dict(
                 cpu=0.25,
                 gpu=0))

    def setUp(self):
        self.folder_in = tempfile.mkdtemp()
        for i in range(10):
            DicommlCase(**sample_case_config(
                caseid='train-case-{i}'.format(i=i),
                n_images=10
            )).save(os.path.join(self.folder_in, 'train'))
        for i in range(4):
            DicommlCase(**sample_case_config(
                caseid='eval-case-{i}'.format(i=i),
                n_images=10
            )).save(os.path.join(self.folder_in, 'eval'))

    def test_task(self):
        import ray
        ray.init(num_cpus=1, num_gpus=0, local_mode=True)
        result = run_task(
            task_class='tasks.tasks.DicommlTrain',
            config=self.get_config())
        self.assertIsInstance(result, dict)

    def test_step_save_load(self):
        from dicomml.tasks.trainable import DicommlTrainable

        trainable = DicommlTrainable(
            config=self.get_config()['trainable_config'])
        _ = trainable.step()
        with tempfile.TemporaryDirectory() as tmp_checkpoint_dir:
            _dir = trainable.save_checkpoint(tmp_checkpoint_dir)
            trainable.load_checkpoint(_dir)

    def tearDown(self):
        shutil.rmtree(self.folder_in)
