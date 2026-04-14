"""
This script is for evaluating the performance of the model on the unseen test set. It calculates metrics such as mIoU (Mean Intersection over Union), Precision, Recall, and F1 Score to assess the model's performance in segmenting the Atmospheric Rivers.
"""

import xarray as xr
import numpy as np
import glob
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepAR model on test dataset."
    )
    parser.add_argument(
        "--pred_dir", type=str, help="Directory containing prediction .nc files."
    )
    parser.add_argument(
        "--true_dir",
        type=str,
        help="Directory containing ground truth (PIKART) .nc files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Probability threshold for binarizing predictions.",
    )
    return parser.parse_args()


def evaluate_ar_model(predict_dir, gt_dir, threshold):
    """
    Evaluates AR detection model using the global accumulator pattern.

    Args:
        predict_dir (str): Directory containing prediction .nc files.
        gt_dir (str): Directory containing ground truth (PIKART) .nc files.
        threshold (float): The probability threshold for binarizing predictions.
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0

    prediction_files = glob.glob(os.path.join(predict_dir, "*_inference.nc"))
    ground_truth_files = glob.glob(os.path.join(gt_dir, "*.nc"))

    prediction_files.sort()
    ground_truth_files.sort()

    print(
        f"Starting evaluation across {len(prediction_files)} files with threshold {threshold}..."
    )

    for pred_file, true_file in zip(prediction_files, ground_truth_files):
        # Load the NetCDF datasets
        ds_pred = xr.open_dataset(pred_file)
        ds_true = xr.open_dataset(true_file)

        # Rename coords to match if necessary
        if 'latitude' in ds_true.coords:
            ds_true = ds_true.rename({'latitude': 'lat', 'longitude': 'lon'})
            
        # Find common timestamps between prediction and ground truth to avoid KeyError
        common_times = np.intersect1d(ds_pred.time.values, ds_true.time.values)
        ds_pred = ds_pred.sel(time=common_times)
        ds_true = ds_true.sel(time=common_times)

        probabilities = ds_pred["ar_mask"].values
        ground_truth = ds_true["ar_mask"].values

        predictions = (probabilities >= threshold).astype(int)

        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"Processed {pred_file} | Yearly TP: {tp}, FP: {fp}, FN: {fn}")

        # Close datasets to free memory
        ds_pred.close()
        ds_true.close()

    # Add a small epsilon to avoid division by zero.
    epsilon = 1e-8

    global_iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    print("-" * 30)
    print("FINAL GLOBAL METRICS")
    print("-" * 30)
    print(f"Global IoU: {global_iou:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")

    return global_iou, precision, recall, f1_score


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.isdir(args.pred_dir):
        print(f"Error: Prediction directory '{args.pred_dir}' does not exist.")
        exit(1)
    if not os.path.isdir(args.true_dir):
        print(f"Error: Ground truth directory '{args.true_dir}' does not exist.")
        exit(1)
    evaluate_ar_model(args.pred_dir, args.true_dir, args.threshold)