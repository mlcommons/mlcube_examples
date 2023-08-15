"""Logic file"""
import os
from shutil import copyfile
from src.utils.utilities import helper


def logic_wrapper(input_folder, output_folder):
    """Edit your logic here"""
    input_file = os.path.normpath(input_folder + "/BraTS_example_seg.nii.gz")
    output_file = os.path.normpath(output_folder + "/BraTS_example_seg.nii.gz")

    # copy paste your logic here
    print("wrapper: Here you can place your own logic")

    # example logic
    copyfile(input_file, output_file)
    helper()


def run_code(input_folder, output_folder, application_name, application_version):
    """Main function"""
    print(
        "*** code execution started:", application_name,
        "version:", application_version, "! ***",
    )

    logic_wrapper(input_folder, output_folder)

    print(
        "*** code execution finished:", application_name,
        "version:", application_version, "! ***",
    )
