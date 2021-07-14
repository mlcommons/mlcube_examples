"""
This requires the MLCube 2.0 that's located somewhere in one of dev branches.
"""
import os
import yaml
from mlcube_docker.docker_run import DockerRun


def main():
    """
    python example_runner.py --mlcube=PATH --platform=docker --task download|train [--workspace=PATH]
        mlcube is the same as root below

    Minimal demo A:
        1. mlcube_config <- load $mlcube/mlcube.yaml
        2. user_config <- If exists, load ${HOME}/.mlcube.yaml
        3. Do something like:
            tmp = mlcube_config.container
            mlcube_config.container = user_config.container
            mlcube_config.container.update(tmp)
        4. Follow the code below - run this one particular task (task).
    """
    root: str = os.path.dirname(os.path.abspath(__file__))
    workspace: str = os.path.join(root, 'workspace')  # or take from a command line

    for task in ('download', 'train'):
        with open(os.path.join(root, 'mlcube.yaml')) as stream:
            mlcube_config = yaml.load(stream.read(), Loader=yaml.SafeLoader)
        docker_runner = DockerRun(mlcube_config, root=root, workspace=workspace, task=task)
        docker_runner.run()


if __name__ == '__main__':
    main()
