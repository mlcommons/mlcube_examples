"""MLCube handler file"""
import os
import subprocess

import typer
import yaml

app = typer.Typer()


class RunTask(object):
    """Run train task Class"""

    @staticmethod
    def run(parameters_file: str, data_dir: str, output_dir: str, mode: str) -> None:
        """Execute test.py script using python"""

        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)

        env = os.environ.copy()
        env.update(parameters)
        env.update(
            {
                "DATA_DIR": data_dir,
                "OUTPUT_DIR": output_dir,
                "MODE": mode
            }
        )

        process = subprocess.Popen("./run.sh", cwd=".", env=env)
        process.wait()



@app.command("train")
def train(
    parameters_file: str = typer.Option(..., "--output_dir"),
    data_dir: str = typer.Option(..., "--output_dir"),
    output_dir: str = typer.Option(..., "--output_dir")
):
    """List test linked to test.py script"""
    RunTask.run(parameters_file, data_dir, output_dir, "train")


@app.command("inference")
def inference(
    parameters_file: str = typer.Option(..., "--output_dir"),
    data_dir: str = typer.Option(..., "--output_dir"),
    output_dir: str = typer.Option(..., "--output_dir")
):
    """List test linked to test.py script"""
    RunTask.run(parameters_file, data_dir, output_dir, "inference")


if __name__ == "__main__":
    app()
