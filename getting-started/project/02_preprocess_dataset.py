"""Preprocess the dataset and save in CSV format"""
import os
import argparse
import pandas as pd

def process_data(input_file, output_file):
    """Process raw dataset and save it in CSV format.
    Args:
        data_dir (str): Folder path containing dataset."""

    col_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"]
    print(f"\nProcessing raw file: {input_file}")

    df = pd.read_csv(input_file, skiprows=22, header=None, delim_whitespace=True)
    df_even=df[df.index%2==0].reset_index(drop=True)
    df_odd=df[df.index%2==1].iloc[: , :3].reset_index(drop=True)
    df_odd.columns = [11,12,13]
    dataset = df_even.join(df_odd)
    dataset.columns = col_names

    dataset.to_csv(output_file, index=False)
    print(f"Processed dataset saved at: {output_file}")


def main():

    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--input_file', required=True,
                        help='Input dataset file')
    parser.add_argument('--output_file', required=True,
                        help='Output processed file')
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    process_data(input_file, output_file)


if __name__ == '__main__':
    main()
