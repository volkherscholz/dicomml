from typing import Union, Dict, List
import logging
import glob
import os

from ray import tune

from dicomml import resolve as dicomml_resolve
from dicomml.log import setup_logging
from dicomml.tasks.trainable import DicommlTrainable


class DicommlTask:
    """
    Base class for Dicomml Tasks,
    may it be data preparation, training
    or test runs
    """

    def __init__(self,
                 name: Union[str, None] = None,
                 config: Union[dict, None] = None):
        if name is None:
            name = type(self).__name__
        self.name = name
        self.config = config or {}
        setup_logging()
        self.logger = logging.getLogger('{}.{}'.format(
            type(self).__module__,
            type(self).__name__))

    def run(self):
        self.logger.info('DicommlTask {} started ...'.format(self.name))
        result = self.task(**self.config)
        self.logger.info('... DicommlTask {} stopped.'.format(self.name))
        return result

    def task(self, **kwargs) -> dict:
        pass


class DicommlLTS(DicommlTask):

    def task(self,
             filenames: Union[List[str], None] = None,
             num_concurrent_files: Union[int, None] = None,
             folder_in: Union[str, None] = None,
             filename_pattern: str = '*.zip',
             load_config: dict = dict(),
             steps: Dict[str, dict] = dict(),
             folder_out: str = './out',
             save_config: dict = dict()):
        from dicomml.transforms.io import Save, Load
        # construct transforms
        transforms = []
        for kind, config in steps.items():
            if isinstance(kind, str):
                Task = dicomml_resolve(kind)
            else:
                Task = kind
            transforms.append(Task(**config))
        # load
        if filenames is None:
            filenames = glob.glob(os.path.join(folder_in, filename_pattern))
        if num_concurrent_files is None:
            num_concurrent_files = len(filenames)

        def shards():
            for i in range(0, len(filenames), num_concurrent_files):
                yield filenames[i:i + num_concurrent_files]

        results = {}
        for files in shards():
            cases = Load(**load_config)(files=files)
            # transform
            for transform in transforms:
                cases = transform(cases)
            # save
            files = Save(**save_config)(cases=cases, folder=folder_out)
            results.update(files)
        return results


class DicommlTrain(DicommlTask):

    def task(self,
             trainable_config: dict,
             metric: str,
             trainable_class: Union[str, tune.Trainable] = DicommlTrainable,
             mode: str = 'min',
             scheduler_class: Union[str, None] = None,
             scheduler_config: Union[dict, None] = None,
             scheduler: Union[
                 tune.schedulers.trial_scheduler.TrialScheduler, None] = None,
             **kwargs):
        if isinstance(trainable_class, str):
            Trainable = dicomml_resolve(trainable_class)
        else:
            Trainable = trainable_class
        if isinstance(scheduler_class, str):
            scheduler = dicomml_resolve(scheduler_class)(**scheduler_config)
        analysis = tune.run(
            run_or_experiment=Trainable,
            config=trainable_config,
            scheduler=scheduler,
            checkpoint_at_end=True,
            **kwargs)
        best_trial = analysis.get_best_trial(metric=metric, mode=mode)
        checkpoint = analysis.get_best_checkpoint(trial=best_trial, mode=mode)
        self.logger.info(
            'Best trial is checkpointed at {checkpoint} \
             and has config {config}'.format(
                checkpoint=checkpoint,
                config=str(best_trial.config)))
        return dict(best_checkpoint=checkpoint)
