"""MLCube handler file"""
import os
import typer
import subprocess


app = typer.Typer()


class EvaluateTask(object):
    """Runs evaluation metrics given the predictions and label files
    Args:
        object ([type]): [description]
    """

    @staticmethod
    def run(
        ground_truth: str, predictions: str, parameters_file: str, output_file: str
    ) -> None:
        cmd = f"python3 metrics.py --ground_truth={ground_truth} --predictions={predictions} --parameters_file={parameters_file} --output_file={output_file}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


@app.command("evaluate")
def evaluate(
    ground_truth: str = typer.Option(..., "--ground_truth"),
    predictions: str = typer.Option(..., "--predictions"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
):
    EvaluateTask.run(ground_truth, predictions, parameters_file, output_path)


@app.command("test")
def test():
    pass


if __name__ == "__main__":
    app()
