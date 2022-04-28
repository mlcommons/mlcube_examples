"""MLCube handler file"""
import os
import typer
import yaml
import subprocess


app = typer.Typer()


class PreprocessTask:
    """Run preprocessing given the input data path"""

    @staticmethod
    def run(
        data_path: str, parameters_file: str, output_path: str
    ) -> None:

        cmd = f"python3 /workspace/preprocess.py --data_path={data_path} \
            --parameters_file {parameters_file} --output_path {output_path}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()

class SanityCheckTask:
    """Run sanity check"""

    @staticmethod
    def run(
        data_path: str, parameters_file: str
    ) -> None:

        cmd = f"python3 sanity_check.py --data_path={data_path}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


class StatisticsTask:
    """Run statistics"""

    @staticmethod
    def run(
        data_path: str, parameters_file: str, output_path: str
    ) -> None:

        cmd = f"python3 statistics.py --data_path={data_path} --out_file={output_path}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


@app.command("prepare")
def prepare(
    data_path: str = typer.Option(..., "--data_path"),
    labels_path: str = typer.Option(..., "--labels_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
):
    PreprocessTask.run(data_path, parameters_file, output_path)


@app.command("sanity_check")
def sanity_check(
    data_path: str = typer.Option(..., "--data_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
):
    SanityCheckTask.run(data_path, parameters_file)

@app.command("statistics")
def statistics(
    data_path: str = typer.Option(..., "--data_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path")
):
    StatisticsTask.run(data_path, parameters_file, output_path)


if __name__ == "__main__":
    app()
