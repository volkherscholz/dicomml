#! /usr/bin/python

#################################
#
# Training with hp optimization
#
#################################

import argparse
import os
import numpy as np

import ray
from ray import tune

from dicomml.tasks.main import run_task


def get_arguments():
    parser = argparse.ArgumentParser(prog='dicomml')
    parser.add_argument(
        "--folder_in",
        action='store',
        dest='folder_in',
        help='input folder')
    return parser.parse_args()


def get_config(args) -> dict:
    scheduler = tune.schedulers.PopulationBasedTraining(
        time_attr="training_step_count",
        metric='BinaryAccuracy',
        mode='max',
        perturbation_interval=10,
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
            train_iterations_per_step=10,
            eval_iterations_per_step=10,
            train_path=os.path.join(args.folder_in, 'train', '*.zip'),
            eval_path=os.path.join(args.folder_in, 'eval', '*.zip'),
            train_batch_size=10,
            eval_batch_size=10,
            shuffle_buffer_size=100,
            data_keys=['images', 'truth'],
            padded_shapes={'images': [10, 300, 300, 1], 'truth': [10, 300, 300, 1]},  # noqa 501
            transformations={
                'transforms.array.Window': dict(window='soft_tissue')},
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
        stop={"training_iteration": 1000},
        num_samples=4,
        reuse_actors=True,
        resources_per_trial=dict(
            cpu=2,
            gpu=0.25))


def main():
    args = get_arguments()
    config = get_config(args)
    ray.init(num_cpus=8, num_gpus=1)
    result = run_task(
        task_class='tasks.tasks.DicommlTrain',
        config=config)
    ray.shutdown()
    print(result)


if __name__ == "main":
    main()
