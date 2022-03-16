"""MLCube handler file"""
import os
import typer
import subprocess


app = typer.Typer()


class PreprocessTask:
    """Runs preprocessing given the input data path"""

    @staticmethod
    def run(
        data_path: str, parameters_file: str, output_path: str
    ) -> None:

        env = os.environ.copy()
        env.update({
            'data_path': data_path,
            'parameters_file': parameters_file,
            'output_path': output_path
        })

        process = subprocess.Popen("./run.sh", cwd=".", env=env)
        process.wait()


@app.command("prepare")
def prepare(
    data_path: str = typer.Option(..., "--data_path"),
    labels_path: str = typer.Option(..., "--labels_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
):
    PreprocessTask.run(data_path, parameters_file, output_path)


@app.command("test")
def test():
    pass


if __name__ == "__main__":
    app()
