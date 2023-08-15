"""Metrics file"""
import os
import argparse
import glob
import yaml
import numpy as np
import nibabel as nib
from shutil import copyfile
from tqdm import tqdm


def preprocess(image: np.ndarray):
    """Preprocess the image labels from a numpy array"""

    image_WT = image.copy()
    image_WT[image_WT == 1] = 1
    image_WT[image_WT == 2] = 1
    image_WT[image_WT == 4] = 1

    image_TC = image.copy()
    image_TC[image_TC == 1] = 1
    image_TC[image_TC == 2] = 0
    image_TC[image_TC == 4] = 1

    image_ET = image.copy()
    image_ET[image_ET == 1] = 0
    image_ET[image_ET == 2] = 0
    image_ET[image_ET == 4] = 1

    image = np.stack([image_WT, image_TC, image_ET])
    image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))

    return image


def load_img(file_path):
    """Reads segmentations image as a numpy array"""

    data = nib.load(file_path)
    data = np.asarray(data.dataobj)
    return data


def get_data_arr(data_path):
    """Reads the content for the data path folder
    and then returns the data in numpy array format"""

    image_path_list = glob.glob(data_path + "/*")
    images_arr = []
    for image_path in image_path_list:
        image = load_img(image_path)
        image = preprocess(image)
        images_arr.append(image)
    images_arr = np.concatenate(images_arr)
    return images_arr


def save_processed_data(output_path, output_filename, images_arr):
    """Writes processed images to the target output path"""
    output_file_path = os.path.join(output_path, output_filename)
    with open(output_file_path, 'wb') as f:
        np.save(f, images_arr)
    print("File correctly saved!")


def main():
    """Main function that recieves input data and preprocess it"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "--data-path",
        type=str,
        required=True,
        help="Directory containing input data",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        type=str,
        required=True,
        help="Path where output data will be stored",
    )
    parser.add_argument(
        "--parameters_file",
        "--parameters-file",
        type=str,
        required=True,
        help="File containing parameters for processing",
    )
    args = parser.parse_args()

    with open(args.parameters_file, "r") as f:
        params = yaml.full_load(f)

    images_arr = get_data_arr(args.data_path)
    save_processed_data(args.output_path, params["output_filename"], images_arr)

    input_file = os.path.normpath(args.data_path + "/BraTS_example_seg.nii.gz")
    output_file = os.path.normpath(args.output_path + "/BraTS_example_seg.nii.gz")
    copyfile(input_file, output_file)

if __name__ == "__main__":
    main()
