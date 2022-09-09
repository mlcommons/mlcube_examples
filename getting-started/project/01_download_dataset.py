"""Download the raw Boston Housing Dataset"""
import os
import yaml
import argparse
import requests

def download_dataset(data_dir, dataset_url):
    """Download dataset and store it in a given path.
    Args:
        data_dir (str): Dataset download path."""

    request = requests.get(dataset_url)
    file_name = "raw_dataset.txt"
    file_path = os.path.join(data_dir, file_name)
    with open(file_path,'wb') as f:
        f.write(request.content)
    print(f"\nRaw dataset saved at: {file_path}")


def main():

    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('--data_dir', required=True,
                        help='Dataset download path')
    parser.add_argument('--parameters_file', required=True,
                        help='File containing extra parameters')
    args = parser.parse_args()

    data_dir = args.data_dir
    parameters_file = args.parameters_file

    with open(parameters_file, 'r') as stream:
        parameters = yaml.safe_load(stream)
    
    dataset_url = parameters["dataset_url"]

    download_dataset(data_dir, dataset_url)


if __name__ == '__main__':
    main()
