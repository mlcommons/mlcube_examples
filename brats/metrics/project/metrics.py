"""Logic file"""
import argparse
import glob
import yaml
from pkgutil import get_data
import nibabel as nib
import numpy as np


def dice_coef_metric(
    probabilities: np.ndarray, truth: np.ndarray, treshold: float = 0.5, eps: float = 0
) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = probabilities >= treshold
    assert predictions.shape == truth.shape
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(
    probabilities: np.ndarray, truth: np.ndarray, treshold: float = 0.5, eps: float = 0
) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = probabilities >= treshold
    assert predictions.shape == truth.shape

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def preprocess_mask_labels(mask: np.ndarray):

    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1

    mask_TC = mask.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 4] = 1

    mask_ET = mask.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 4] = 1

    mask = np.stack([mask_WT, mask_TC, mask_ET])
    mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

    return mask


def load_img(file_path):
    data = nib.load(file_path)
    data = np.asarray(data.dataobj)
    return data


def get_data_arr(predictions_path, ground_truth_path):
    predictions = glob.glob(predictions_path + "/*")
    ground_truth = glob.glob(ground_truth_path + "/*")
    if not len(predictions) == len(ground_truth):
        raise ValueError(
            "Number of predictions should be the same of ground truth labels"
        )
    gt_arr, prediction_arr = [], []
    for gt_path, prediction_path in zip(ground_truth, predictions):
        gt = load_img(gt_path)
        gt = preprocess_mask_labels(gt)
        prediction = load_img(prediction_path)
        prediction = preprocess_mask_labels(prediction)
        gt_arr.append(gt)
        prediction_arr.append(prediction)
    gt_arr = np.concatenate(gt_arr)
    prediction_arr = np.concatenate(prediction_arr)
    return gt_arr, prediction_arr


def create_metrics_file(output_file, results):
    with open(output_file, "w") as f:
        yaml.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Directory containing the ground truth data",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Directory containing the predictions",
    )
    parser.add_argument(
        "--output_file",
        "--output-file",
        type=str,
        required=True,
        help="file to store metrics results as YAML",
    )
    parser.add_argument(
        "--parameters_file",
        "--parameters-file",
        type=str,
        required=True,
        help="File containing parameters for evaluation",
    )
    args = parser.parse_args()

    with open(args.parameters_file, "r") as f:
        params = yaml.full_load(f)

    gt_arr, pred_arr = get_data_arr(args.predictions, args.ground_truth)

    treshold = float(params["treshold"])
    eps = float(params["eps"])

    dice_coef = dice_coef_metric(pred_arr, gt_arr, treshold, eps)
    jaccard_coef = jaccard_coef_metric(pred_arr, gt_arr, treshold, eps)

    results = {
        "dice_coef": str(dice_coef),
        "jaccard_coef": str(jaccard_coef),
    }

    print(results)
    create_metrics_file(args.output_file, results)


if __name__ == "__main__":
    main()
