""" Example entry point script compatible with MLCube protocol. """
import argparse
from enum import Enum
from typing import List


class Task(str, Enum):
    """ Every task has a name. This example defines two tasks - `task_a` and `task_b`.  """
    TASK_A = 'task_a'
    TASK_B = 'task_b'


def task_a(task_args: List[str]) -> None:
    # Parse task-specific command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task_a_arg_01', type=str, default=None, help="Argument 01 for task A.")
    # parser.add_argument('--task_a_arg_02', type=str, default=None, help="Argument 02 for task A.")
    # args = parser.parse_args(args=task_args)

    # Implement this task here
    ...


def task_b(task_args: List[str]) -> None:
    # Parse task-specific command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task_b_arg_01', type=str, default=None, help="Argument 01 for task B.")
    # parser.add_argument('--task_b_arg_02', type=str, default=None, help="Argument 02 for task B.")
    # args = parser.parse_args(args=task_args)

    # Implement this task here
    ...


def main():
    # Every MLCuber runner passes a task name as the first argument. Other arguments are task-specific.
    parser = argparse.ArgumentParser()
    parser.add_argument('mlcube_task', type=str, help="Task for this MLCube.")

    # The `mlcube_args` contains task name (mlcube_args.mlcube_task)
    # The `task_args` list needs to be parsed later when task name is known
    mlcube_args, task_args = parser.parse_known_args()

    if mlcube_args.mlcube_task == Task.TASK_A:
        task_a(task_args)
    if mlcube_args.mlcube_task == Task.TASK_B:
        task_b(task_args)
    else:
        raise ValueError(f"Unknown task: '{mlcube_args.mlcube_task}'")


if __name__ == '__main__':
    main()
