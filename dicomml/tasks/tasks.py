from typing import Union, Dict
import logging

from ray import tune

from dicomml import resolve as dicomml_resolve
from dicomml.logging import setup_logging


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
            type(self).__module__.__name__,
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
             folder_in: str = '.',
             load_config: dict = dict(),
             steps: Dict[str, dict] = dict(),
             folder_out: str = './out',
             save_config: dict = dict()):
        from dicomml.transforms.io import Save, Load
        # construct transforms
        transforms = []
        for kind, config in steps.items():
            transforms.append(dicomml_resolve(kind)(**config))
        # load
        cases = Load(**load_config)(folder=folder_in)
        # transform
        for transform in transforms:
            cases = transform(cases)
        # save
        files = Save(**save_config)(cases=cases, folder=folder_out)
        return files


class DicommlTrain(DicommlTask):

    def task(self,
             trainable_class: Union[str, tune.Trainable],
             trainable_config: dict,
             metric: str,
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
        if isinstance(scheduler_class):
            scheduler = dicomml_resolve(scheduler_class)(**scheduler_config)
        analysis = tune.run(
            run_or_experiment=Trainable,
            config=trainable_config,
            scheduler=scheduler,
            mode=mode,
            **kwargs)
        best_trial = analysis.get_best_trial(metric=metric, mode=mode)
        checkpoint = analysis.get_best_checkpoint(trial=best_trial, mode=mode)
        self.logger.info(
            'Best trial is checkpointed at {checkpoint} \
             and has config {config}'.format(
                checkpoint=checkpoint,
                config=str(best_trial.config)))
        return dict(best_checkpoint=checkpoint)
