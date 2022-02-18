"""MLCube handler file"""
import os
import typer
import yaml
from src.my_logic import run_code

app = typer.Typer()


class ExampleTask(object):
    """Example task Class
    It reads the content of the parameters file and then
    prints "MY_NEW_PARAMETER_EXAMPLE"."""

    @staticmethod
    def run_example(parameters_file: str) -> None:

        # Load parameters from the paramters file
        with open(parameters_file, "r") as stream:
            parameters = yaml.safe_load(stream)

        print("This is my new parameter example:")
        print(parameters["MY_NEW_PARAMETER_EXAMPLE"])


class InferTask(object):
    """Run task Class
    It defines the environment variables:
        data_path: Directory path to dataset
        output_path: Directory path to final results
        All other parameters are defined in parameters_file
    Then executes the run_code method inside my_logic script"""

    @staticmethod
    def run(data_path: str, output_path: str, parameters_file: str) -> None:

        # Load parameters from the paramters file
        with open(parameters_file, "r") as stream:
            parameters = yaml.safe_load(stream)

        application_name = parameters["APPLICATION_NAME"]
        application_version = parameters["APPLICATION_VERSION"]
        run_code(data_path, output_path, application_name, application_version)


@app.command("example")
def example(parameters_file: str = typer.Option(..., "--parameters_file")):
    ExampleTask.run_example(parameters_file)


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),
    parameters_file: str = typer.Option(..., "--parameters_file")
):
    InferTask.run(data_path, output_path, parameters_file)


if __name__ == "__main__":
    app()
