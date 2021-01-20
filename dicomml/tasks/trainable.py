import logging
import os
import tempfile
from typing import Tuple, Dict, List, Union

from ray import tune
import torch

from dicomml import resolve as dicomml_resolve
from dicomml.log import setup_logging
from dicomml.cases.case import DicommlCase
from dicomml.transforms import DicommlTransform


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
        self.loss_metric.reset_states()
        for metric in self.eval_metrics:
            metric.reset_states()
        # training steps
        for _ in range(self.train_iterations_per_step):
            self.train_step(**next(self.train_dataset))
        # evaluation steps
        for _ in range(self.eval_iterations_per_step):
            self.eval_step(**next(self.eval_dataset))
        # return evaluation metric results & loss
        return dict(
            training_step_count=self.training_step_count.read_value().numpy(),
            epoch=self.iteration,
            loss=self.loss_metric.result().numpy(),
            **{type(metric).__name__: metric.result().numpy()
               for metric in self.eval_metrics})

    def save_checkpoint(self, tmp_checkpoint_dir: str):
        """
        save current state to checkpoint
        """
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "checkpoint")
        torch.save({
            key: var.state_dict()
            for key, var in self.get_variables().items()})
        return checkpoint_path

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
                   shuffle_buffer_size: int = 100,
                   cache_dir: str = '',
                   padded_shapes: Union[list, None] = None,
                   data_keys: List[str] = [],
                   export_config: dict = dict(),
                   **kwargs):
        import tensorflow as tf

        class _Dataset(torch.utils.data.Dataset):

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
                return self.export(
                    self.apply_transforms(
                        self.load(
                            self.files[idx])))

            def load(filename):
                return [DicommlCase.load(filename)]

            def apply_transforms(cases):
                for transform in self.transformations:
                    cases = transform(cases)
                return cases

            def export(cases):
                # assume that all transformations only
                # act on one case
                return cases[0].export(**self.export_kwargs)

        class _DicommlDataset(tf.data.Dataset):

            @classmethod
            def from_folder(cls,
                            path: str = '.',
                            data_keys: List[str] = [],
                            transformations: List[DicommlTransform] = [],
                            shuffle_buffer_size: int = 10,
                            batch_size: int = 1,
                            cache_file: str = '',
                            padded_shapes: Union[list, None] = None,
                            padding_value: Union[float, None] = None,
                            drop_remainder: bool = False,
                            **kwargs) -> '_DicommlDataset':
                import glob
                files = [
                    _f for _f in glob.glob(path, recursive=True)
                    if os.path.isfile(_f)]

                def load_case_transform(file):

                    def load(file):
                        return [DicommlCase.load(file.numpy().decode('utf-8'))]

                    def apply_transforms(cases):
                        for transform in transformations:
                            cases = transform(cases)
                        return cases

                    def export(cases):
                        return [[
                            case.export(**kwargs)[data_key]
                            for data_key in data_keys]
                            for case in cases]

                    func = tf.py_function(
                        func=lambda file: export(apply_transforms(load(file))),
                        inp=[file],
                        Tout=[tf.float32 for _ in range(len(data_keys))])

                    return cls.from_tensor_slices({
                        data_key: func[i]
                        for i, data_key in enumerate(data_keys)})

                # construct datasets
                return cls.from_tensor_slices(files) \
                    .interleave(
                        lambda _file: load_case_transform(_file),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False) \
                    .padded_batch(
                        batch_size=batch_size,
                        padded_shapes=padded_shapes,
                        padding_values=padding_value,
                        drop_remainder=drop_remainder) \
                    .cache(filename=cache_file) \
                    .prefetch(tf.data.experimental.AUTOTUNE) \
                    .shuffle(shuffle_buffer_size) \
                    .repeat()

        transformations = [
            dicomml_resolve(kind)(**cfg)
            for kind, cfg in transformations.items()]
        if cache_dir != '':
            os.makedirs(cache_dir, exist_ok=True)
            train_cache = os.path.join(cache_dir, 'train')
            eval_cache = os.path.join(cache_dir, 'eval')
        else:
            train_cache = ''
            eval_cache = ''

        self.train_dataset = iter(_DicommlDataset.from_folder(
            path=train_path,
            transformations=transformations,
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=train_batch_size,
            cache_file=train_cache,
            data_keys=data_keys,
            padded_shapes=padded_shapes,
            **export_config))
        self.eval_dataset = iter(_DicommlDataset.from_folder(
            path=eval_path,
            transformations=transformations,
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=eval_batch_size,
            cache_file=eval_cache,
            data_keys=data_keys,
            padded_shapes=padded_shapes,
            **export_config))

    def setup_training(self,
                       loss_function: Tuple[str, dict],
                       train_metric: Tuple[str, dict],
                       eval_metrics: Dict[str, dict],
                       model_class: str,
                       optimizer_class: str,
                       **kwargs):
        import tensorflow as tf
        # setup metrics
        _class, _config = loss_function
        self.loss_function = dicomml_resolve(_class, prefix='tf')(**_config)
        _class, _config = train_metric
        self.loss_metric = dicomml_resolve(_class, prefix='tf')(**_config)
        self.eval_metrics = [
            dicomml_resolve(kind, prefix='tf')(**config)
            for kind, config in eval_metrics.items()]
        # ensure that the variables are defined
        try:
            # make sure not to reset trainstep if config is reset,
            # i.e. in case of actor reuse
            getattr(self, 'training_step_count')
        except AttributeError:
            self.training_step_count = tf.Variable(
                initial_value=0, dtype=tf.int64, trainable=False)
        self.model = dicomml_resolve(model_class)(**{
            key[6:]: val for key, val in kwargs.items()
            if 'model_' in key})
        self.optimizer = dicomml_resolve(optimizer_class, prefix='tf')(**{
            key[10:]: val for key, val in kwargs.items()
            if 'optimizer_' in key})

        # train step
        @tf.function
        def train_step(images, truth):
            with tf.GradientTape() as tape:
                # run the inputs through the model
                # record operations
                predictions = self.model(images)
                # compute loss
                loss_value = self.loss_function(predictions, truth)
                # add layer regularization losses
                loss_value += sum(self.model.losses)
            # Obtain gradients
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            # Apply gradients
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))
            # update metrics
            self.loss_metric.update_state(predictions, truth)
            self.training_step_count.assign_add(1)

        # eval step
        @tf.function
        def eval_step(images, truth):
            predictions = self.model(images)
            for metric in self.eval_metrics:
                metric.update_state(predictions, truth)

        self.train_step = train_step
        self.eval_step = eval_step

    def get_variables(self) -> dict:
        return dict(
            model=self.model,
            optimizer=self.optimizer)
