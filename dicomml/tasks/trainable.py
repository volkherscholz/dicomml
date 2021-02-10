import logging
import os
import tempfile
from typing import Tuple, Dict
from functools import partial
import numpy as np

from ray import tune
import torch

from dicomml import resolve as dicomml_resolve
from dicomml.log import setup_logging
from dicomml.cases.case import DicommlCase


class DicommlTrainable(tune.Trainable):
    """
    Subclass of tune.Trainable for training and
    hyperparameter optimization.
    Implements training of DicommlModels on DicommlDatasets
    """

    def setup(self, config: dict):
        setup_logging(**config.get('logging', dict()))
        self.logger = logging.getLogger('{module}.{classname}'.format(
            module=type(self).__module__,
            classname=type(self).__name__))
        # setup some basis stuff
        self.checkpoints_path = config.get(
            'checkpoint_path',
            os.path.join(tempfile.gettempdir(), type(self).__name__))
        self.train_iterations_per_step = config.get(
            'train_iterations_per_step', 10)
        self.eval_iterations_per_step = config.get(
            'eval_iterations_per_step', 10)
        self.device = "cuda:0" if (
            config.get("use_gpu", True) and torch.cuda.is_available()
            ) else "cpu"
        self.metric_states = _MetricStates()
        try:
            self.setup_data(**config)
            self.setup_training(**config)
        except ValueError as e:
            self.logger.error('Error setting up the training', exc_info=e)
            raise e

    def reset_config(self, new_config):
        try:
            self.setup_training(**new_config)
            self.config = new_config
            return True
        except ValueError as e:
            self.logger.error('Error resetting the training', exc_info=e)
            raise e

    def step(self):
        # reset metrics
        self.metric_states.reset()
        # training steps
        _loss_values = []
        for _ in range(self.train_iterations_per_step):
            try:
                data = next(self.train_dataiter)
            except StopIteration:
                self.train_dataiter = iter(self.train_dataloader)
                data = next(self.train_dataiter)
            _loss_val = self.train_step(**data)
            _loss_values.append(_loss_val)
        # evaluation steps
        for _ in range(self.eval_iterations_per_step):
            try:
                data = next(self.eval_dataiter)
            except StopIteration:
                self.eval_dataiter = iter(self.eval_dataloader)
                data = next(self.eval_dataiter)
            self.metric_states(self.eval_step(**data))
        # return evaluation metric results & loss
        return dict(
            epoch=self.iteration,
            training_loss=max(_loss_values),
            **self.metric_states.result())

    def save_checkpoint(self, tmp_checkpoint_dir: str):
        """
        save current state to checkpoint
        """
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        torch.save({
            key: var.state_dict()
            for key, var in self.get_variables().items()},
            checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir: str):
        """
        Initialize or reload from checkpoints
        """
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        checkpoint = torch.load(checkpoint_path)
        for key, var in self.get_variables().items():
            var.load_state_dict(checkpoint[key])
        self.logger.info('Restored from checkpoint {}'.format(
            tmp_checkpoint_dir))

    def setup_data(self,
                   train_path: str = os.path.join('.', 'data', 'train'),
                   eval_path: str = os.path.join('.', 'data', 'eval'),
                   train_batch_size: int = 10,
                   eval_batch_size: int = 10,
                   transformations: Dict[str, dict] = dict(),
                   num_workers: int = 0,
                   export_config: dict = dict(),
                   **kwargs):
        class _DicommlDataset(torch.utils.data.Dataset):

            def __init__(self,
                         path: str = '.',
                         transformations: dict = dict(),
                         **kwargs):
                import glob
                self.files = [
                    _f for _f in glob.glob(path, recursive=True)
                    if os.path.isfile(_f)]
                self.transformations = [
                    dicomml_resolve(kind)(**cfg)
                    for kind, cfg in transformations.items()]
                self.export_kwargs = kwargs

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()
                return self.export(self.apply_transforms(self.load(idx)))

            def load(self, idx):
                return [DicommlCase.load(self.files[idx])]

            def apply_transforms(self, cases):
                for transform in self.transformations:
                    cases = transform(cases)
                return cases

            def export(self, cases):
                # assume that all transformations only
                # act on one case
                return cases[0].export(**self.export_kwargs)

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=_DicommlDataset(
                path=train_path,
                transformations=transformations,
                **export_config),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers)
        self.train_dataiter = iter(self.train_dataloader)
        self.eval_dataloader = torch.utils.data.DataLoader(
            dataset=_DicommlDataset(
                path=eval_path,
                transformations=transformations,
                **export_config),
            batch_size=eval_batch_size,
            shuffle=True,
            num_workers=0)
        self.eval_dataiter = iter(self.eval_dataloader)

    def setup_training(self,
                       loss_function: Tuple[str, dict],
                       eval_metrics: Dict[str, dict],
                       model_class: str,
                       optimizer_class: str,
                       model_config: dict = dict(),
                       prediction_target: str = 'class',
                       treshold_value: float = 0.5,
                       **kwargs):
        # setup metrics
        _class, _config = loss_function
        self.loss = dicomml_resolve(_class, prefix='torch')(**_config)
        self.eval_metrics = {
            key: partial(dicomml_resolve(key, prefix='sklearn.metrics'), **cfg)
            for key, cfg in eval_metrics.items()}
        # setup model
        self.model = dicomml_resolve(model_class)(**{
            **model_config,
            **{k[6:]: v for k, v in kwargs.items() if 'model_' in k}})
        self.model.to(self.device)
        # setup optimizer
        self.optimizer = dicomml_resolve(optimizer_class, prefix='torch')(
            self.model.parameters(),
            **{key[10:]: val for key, val in kwargs.items()
               if 'optimizer_' in key})
        # setup logits transformation layer
        if prediction_target == 'class':
            self.logits_layer = torch.nn.Softmax(dim=1)
            self.predict = lambda prob: np.argmax(prob, axis=1)
        elif prediction_target == 'labels':
            self.logits_layer = torch.nn.Sigmoid()
            self.predict = lambda prob: np.where(prob > treshold_value, 0, 1)
        if prediction_target == 'class_probabilities':
            self.logits_layer = torch.nn.Softmax(dim=1)
            self.predict = lambda prob: prob
        elif prediction_target == 'label_probabilities':
            self.logits_layer = torch.nn.Sigmoid()
            self.predict = lambda prob: prob

    def train_step(self,
                   images: torch.Tensor,
                   truth: torch.Tensor):
        images, truth = images.to(self.device), truth.long().to(self.device)
        # forward + backward + optimize
        logits = self.model(images)
        # zero gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        logits = self.model(images)
        loss_value = self.loss(logits, truth)
        loss_value.backward()
        self.optimizer.step()
        return loss_value.detach().item()

    def eval_step(self,
                  images: torch.Tensor,
                  truth: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            images_dev, truth_dev = \
                images.to(self.device), truth.long().to(self.device)
            logits = self.model(images_dev)
            loss_value = self.loss(logits, truth_dev)
            # transform logits
            probabilities = self.logits_layer(logits).cpu().numpy()
        predictions = self.predict(probabilities)
        return {
            'validation_loss': loss_value.item(),
            **{name: metric(
                truth.cpu().numpy().reshape(-1),
                predictions.reshape(-1))
               for name, metric in self.eval_metrics.items()}}

    def get_variables(self) -> dict:
        return dict(
            model=self.model,
            optimizer=self.optimizer)


class _MetricStates:

    def __init__(self):
        self.buffer = {}

    def __call__(self, values: dict):
        for key, val in values.items():
            if key in self.buffer.keys():
                self.buffer[key].append(val)
            else:
                self.buffer.update({key: [val]})

    def reset(self):
        self.buffer = {}

    def result(self, mode='mean'):
        if mode == 'mean':
            result = {
                key: sum(values) / len(values)
                for key, values in self.buffer.items()}
        elif mode == "max":
            result = {
                key: max(values)
                for key, values in self.buffer.items()}
        elif mode == "min":
            result = {
                key: min(values)
                for key, values in self.buffer.items()}
        else:
            result = 0.0
        return result
