import argparse
import numpy as np
import tensorflow as tf
from typing import List
from datetime import datetime
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def matmul(shape_a: List[int], shape_b: List[int], output_file: str) -> None:

    a = tf.random.normal(shape=shape_a)
    b = tf.random.normal(shape=shape_b)
    print(f"shape_a={shape_a}, shape_b={shape_b}")

    start_time = datetime.now()

    x = tf.matmul(a, b)

    print("\n" * 5)
    print("Time taken:", datetime.now() - start_time)
    print("\n" * 5)

    np.savetxt(output_file, x)


if __name__ == '__main__':
    """
    MLCube declares the following contract:
        1. First command line argument is always a task name
        2. Second, third and so on are the task specific parameters. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mlcube_task', type=str, help="Task for this MLCube.")
    parser.add_argument('--parameters_file', '--parameters-file', type=str, required=True,
                        help="YAML file with matrix shapes.")
    parser.add_argument('--output_file', '--output-file', type=str, required=True,
                        help="File with matrix multiplication results.")
    args = parser.parse_args()

    if args.mlcube_task != 'matmul':
        raise ValueError(f"Invalid task: {args.mlcube_task}")

    with open(args.parameters_file) as stream:
        config = yaml.load(stream.read(), Loader=Loader)

    matmul(config['matrix_a'], config['matrix_b'], args.output_file)
