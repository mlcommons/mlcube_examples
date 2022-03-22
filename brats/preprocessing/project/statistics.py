import os
import yaml
import argparse


def get_statistics(data_path):
    """Computes statistics about the data. This statistics are uploaded
    to the Medperf platform under the data owner's approval. Include
    every statistic you consider useful for determining the nature of the
    data, but keep in mind that we want to keep the data as private as 
    possible.
    """

    len_data = len(os.listdir(data_path))

    stats = {
        "data length": len_data
    }

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MedPerf Statistics Example")
    parser.add_argument(
        "--data_path",
        type=str,
        help="directory containing the prepared data",
    )
    parser.add_argument(
        "--out_file", dest="out_file", type=str, help="file to store statistics"
    )

    args = parser.parse_args()

    stats = get_statistics(args.data_path)

    with open(args.out_file, "w") as f:
        yaml.dump(stats, f)
