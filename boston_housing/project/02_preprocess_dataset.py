"""Preprocess the dataset and save in CSV format"""
import os
import argparse
import pandas as pd

def process_data(data_dir):
    """Process raw dataset and save it in CSV format.
    Args:
        data_dir (str): Folder path containing dataset."""

    col_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"]
    raw_file = os.path.join(data_dir, "raw_dataset.txt")
    print(f"\nProcessing raw file: {raw_file}")

    df = pd.read_csv(raw_file, skiprows=22, header=None, delim_whitespace=True)
    df_even=df[df.index%2==0].reset_index(drop=True)
    df_odd=df[df.index%2==1].iloc[: , :3].reset_index(drop=True)
    df_odd.columns = [11,12,13]
    dataset = df_even.join(df_odd)
    dataset.columns = col_names

    output_file = os.path.join(data_dir, "processed_dataset.csv")
    dataset.to_csv(output_file, index=False)
    print(f"Processed dataset saved at: {output_file}")


def main():

    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--data_dir', required=True,
                        help='Folder containing dataset file')
    args = parser.parse_args()

    data_dir = args.data_dir
    process_data(data_dir)


if __name__ == '__main__':
    main()