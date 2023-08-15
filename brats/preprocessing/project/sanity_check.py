"""Sanity check logic"""
import os
import argparse


def sanity_check(data):
    """Runs a few checks to ensure data quality and integrity
    Args:
        names_df (pd.DataFrame): DataFrame containing transformed data.
    """
    # Here you must add all the checks you consider important regarding the
    # state of the data
    assert len(data) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Medperf Model Sanity Check Example")
    parser.add_argument(
        "--data_path",
        dest="data",
        type=str,
        help="directory containing the prepared data",
    )

    args = parser.parse_args()

    data = os.listdir(args.data)
    sanity_check(data)
