#! /usr/bin/python

#################################
#
# Prepare cases for training
#
#################################

import argparse
from dicomml.tasks.main import run_pipeline


def get_arguments():
    parser = argparse.ArgumentParser(prog='dicomml')
    parser.add_argument(
        "--folder_in",
        action='store',
        dest='folder_in',
        help='input folder')
    parser.add_argument(
        "--folder_out",
        action='store',
        dest='folder_out',
        help='output folder')
    return parser.parse_args()


def get_config(args) -> dict:
    return dict(
        folder_in=args.folder_in,
        load_config=dict(
            filename_pattern='*.zip'),
        steps={
            'transforms.expand.Split': dict(
                n_images=10),
            'transforms.expand.AddTranforms': dict(
                base_transform='transforms.array.Rotate',
                n_applications=20,
                value_ranges=dict(angle=(-20, 20)))},
        folder_out=args.folder_out,
        save_config=dict(
            split_ratios=dict(
                train=0.8,
                eval=0.1,
                test=0.1)))


def get_parallel_config():
    return [
        dict(load_config=dict(filename_pattern='{i}*.zip'.format(i=i)))
        for i in range(5)]


def main():
    args = get_arguments()
    config = get_config(args)
    parallel_config = get_parallel_config()
    run_pipeline(tasks=dict(lts=dict(
        task_class='tasks.tasks.DicommlLTS',
        config=config,
        parallel_configs=parallel_config)))


if __name__ == "main":
    main()
