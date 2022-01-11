"""Logic file"""
import os
from shutil import copyfile
from dotenv import dotenv_values
from src.utils.utilities import helper

config = dotenv_values(".env")
APPLICATION_NAME = config["APPLICATION_NAME"]
APPLICATION_VERSION = config["APPLICATION_VERSION"]
INPUT_FOLDER = config["INPUT_FOLDER"]
OUTPUT_FOLDER = config["OUTPUT_FOLDER"]


def logic_wrapper(args):
    """Edit your logic here"""
    if args.data_dir is not None:
        INPUT_FOLDER = args.data_dir
    input_file = os.path.normpath(INPUT_FOLDER + "/something_t1.nii.gz")
    # ... do the same for t1c, flair and t2 here
    if args.results_dir is not None:
        OUTPUT_FOLDER = args.results_dir
    output_file = os.path.normpath(OUTPUT_FOLDER + "/something_seg.nii.gz")

    # copy paste your amazing logic here
    print("wrapper: I can feel the magic happening..it feels like a little sun rising inside me!")

    # example logic
    copyfile(input_file, output_file)
    helper()


def run_code(args):
    """Main function"""
    print("*** code execution started:",
          APPLICATION_NAME, "version:", APPLICATION_VERSION, "! ***")
    logic_wrapper(args)
    print("*** code execution finished:",
          APPLICATION_NAME, "version:", APPLICATION_VERSION, "! ***")
