"""Download the raw Boston Housing Dataset"""
import os
import argparse
import requests

DATASET_URL = "http://lib.stat.cmu.edu/datasets/boston"


def download_dataset(data_dir):
    """Download dataset and store it in a given path.
    Args:
        data_dir (str): Dataset download path."""

    request = requests.get(DATASET_URL)
    file_name = "raw_dataset.txt"
    file_path = os.path.join(data_dir, file_name)
    with open(file_path,'wb') as f:
        f.write(request.content)
    print(f"\nRaw dataset saved at: {file_path}")


def main():

    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('--data_dir', required=True,
                        help='Dataset download path')
    args = parser.parse_args()

    data_dir = args.data_dir
    download_dataset(data_dir)


if __name__ == '__main__':
    main()
