#! /usr/bin/python

#################################
#
# Prepare cases for training
#
#################################

import os
import glob
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
        "--num_workers",
        action='store',
        type=int,
        dest='num_workers',
        default=4,
        help='Number of workers')
    parser.add_argument(
        "--num_concurrent_files",
        action='store',
        type=int,
        dest='num_concurrent_files',
        default=1,
        help='Number of concurrent files processed per worker')
    parser.add_argument(
        "--folder_out",
        action='store',
        dest='folder_out',
        help='output folder')
    return parser.parse_args()


def get_config(args) -> dict:
    filenames = glob.glob(os.path.join(args.folder_in, '*.zip'))
    n = round(len(filenames) / args.num_workers)
    chunks = [filenames[i:i + n] for i in range(0, len(filenames), n)]

    config = dict(
        num_concurrent_files=args.num_concurrent_files,
        steps={
            'transforms.expand.Split': dict(
                n_images=10),
            'transforms.expand.AddTranforms': dict(
                base_transform='transforms.array.Rotate',
                n_applications=20,
                value_ranges=dict(angle=(-20, 20)))},
        folder_out=args.folder_out,
        save_config=dict(
            shuffle_cases=True,
            split_ratios=dict(
                train=0.8,
                eval=0.1,
                test=0.1)))
    parallel_config = [
        dict(filenames=chunk)
        for chunk in chunks]
    return config, parallel_config


def main():
    args = get_arguments()
    config, parallel_config = get_config(args)
    run_pipeline(tasks=dict(lts=dict(
        task_class='tasks.tasks.DicommlLTS',
        config=config,
        parallel_configs=parallel_config)))


if __name__ == "__main__":
    main()
