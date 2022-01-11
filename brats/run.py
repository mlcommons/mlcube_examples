"""Load code"""
import argparse
from src.amazing_logic import run_code

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Path to dataset.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default=None,
    help="Path to output folder.",
)
args, _unkown = parser.parse_known_args()

# execute
run_code(args)
