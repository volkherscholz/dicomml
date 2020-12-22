from typing import Union, List, Dict

import ray

from dicomml import resolve as dicomml_resolve
from dicomml.tasks.tasks import DicommlTask


def run_task(name: Union[str, None] = None,
             task_class: Union[str, DicommlTask] = DicommlTask,
             config: Union[dict, None] = None,
             parallel_configs: Union[List[dict], None] = None
             ) -> List[dict]:
    if isinstance(task_class, str):
        Task = dicomml_resolve(task_class)
    else:
        Task = task_class
    if parallel_configs is not None:
        parallel_configs = {
            '{name}-{i}'.format(name=name, i=i): {**config, **cfg}
            for i, cfg in enumerate(parallel_configs)}
        tasks = [
            ray.remote(Task).options(name=name).remote(name=name, **cfg)
            for name, cfg in parallel_configs.items()]
        results = ray.get([task.run.remote() for task in tasks])
    else:
        results = [Task(**config).run()]
    return results


def run_pipeline(ray_config: dict = dict(),
                 tasks: Dict[str, dict] = dict()):
    # intiate ray
    ray.init(**ray_config)
    results = [{}]
    for name, config in tasks.items():
        for result in results:
            config.update({**result})
        results = run_task(name=name, **config)
    ray.shutdown()


def run_from_file(path: str = ''):
    import os
    import yaml
    if not os.path.exists(path):
        raise ValueError('Path {} does not exist.'.format(path))
    with open(path, 'r') as _f:
        config = yaml.load(_f, Loader=yaml.SafeLoader)
    run_pipeline(**config)


def main():
    import argparse
    parser = argparse.ArgumentParser('Dicomml')
    parser.add_argument(
        name='config_file',
        action='store',
        required=True,
        help='The path to the config file in yaml format')
    args = parser.parse_args()
    run_from_file(path=args.config_file)
