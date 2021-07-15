"""
This requires the MLCube 2.0 that's located somewhere in one of dev branches.
"""
import os
import yaml
from mlcube_docker.docker_run import DockerRun


def load_config(mlcube_config_path: str, user_config_path: str):
    """Returns dictionary containing MLCube configuration"""
    # Load mlcube config data
    try:
        with open(mlcube_config_path) as stream:
                mlcube_config_data = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    except IOError as exc:
        # If file doesn't exist throw the exception:
        # OSError: {PATH_TO}/mnist/mlcube.yaml: No such file or directory
        raise IOError("%s: %s" % (mlcube_config_path, exc.strerror))

    # Load user config data if file exists
    if os.path.isfile(user_config_path):
        with open(user_config_path) as stream:
                user_config_data = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    else:
        return mlcube_config_data

    # Merge config data
    tmp = mlcube_config_data['container']
    mlcube_config_data['container'] = user_config_data['container']
    mlcube_config_data['container'].update(tmp)
    return mlcube_config_data


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
    home = os.path.expanduser("~")
    root: str = os.path.dirname(os.path.abspath(__file__))
    workspace: str = os.path.join(root, 'workspace')  # or take from a command line

    mlcube_config_path = os.path.join(root, 'mlcube.yaml')
    user_config_path = os.path.join(home, '.mlcube.yaml')
    mlcube_config_data = load_config(mlcube_config_path, user_config_path)

    for task in ('download', 'train'):
        docker_runner = DockerRun(mlcube_config_data, root=root, workspace=workspace, task=task)
        docker_runner.run()


if __name__ == '__main__':
    main()
