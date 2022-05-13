"""Preprocess script"""
import argparse
import pathlib
from shutil import copyfile
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser("FeTS preprocess example script")
    parser.add_argument(
        "--data_path", dest="input", type=str, help="path containing input data"
    )
    parser.add_argument(
        "--parameters_file", dest="params", type=str, help="path extra parameters file"
    )
    parser.add_argument(
        "--output_path", dest="output", type=str, help="path to store processed data"
    )

    args = parser.parse_args()

    # Modify the following lines with the needed preprocessing logic

    with open(args.params, "r") as f:
        params = yaml.full_load(f)

    input_file = pathlib.Path(args.input, params["input_filename"])
    output_file = pathlib.Path(args.output, params["output_filename"])
    copyfile(input_file, output_file)
    print("Data has being processed!")
