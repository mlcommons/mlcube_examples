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


class RunTask(object):
    """Run task Class
    It defines the environment variables:
        input_folder: Directory path to dataset
        output_folder: Directory path to final results
        All other parameters are defined in parameters_file
    Then executes the run_code method inside my_logic script"""

    @staticmethod
    def run(input_folder: str, output_folder: str, parameters_file: str) -> None:

        # Load parameters from the paramters file
        with open(parameters_file, "r") as stream:
            parameters = yaml.safe_load(stream)

        application_name = parameters["APPLICATION_NAME"]
        application_version = parameters["APPLICATION_VERSION"]
        run_code(input_folder, output_folder, application_name, application_version)


@app.command("example")
def example(parameters_file: str = typer.Option(..., "--parameters_file")):
    ExampleTask.run_example(parameters_file)


@app.command("run")
def run(
    input_folder: str = typer.Option(..., "--input_folder"),
    output_folder: str = typer.Option(..., "--output_folder"),
    parameters_file: str = typer.Option(..., "--parameters_file")
):
    RunTask.run(input_folder, output_folder, parameters_file)


if __name__ == "__main__":
    app()
