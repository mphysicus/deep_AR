"""
This code is for running the inference on CMIP6 data using the DeepAR model. It loads the trained model, processes the input data, and saves the predictions in NetCDF format. The script is designed to work efficiently with multiple GPUs using the Accelerate library.
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import torch

# Add the 'model' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model")))

from deep_ar import deep_ar_model_registry
from deep_ar.data.datasets import ARInferenceDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for running inference on CMIP6 data using the DeepAR model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained DeepAR model weights.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CMIP6 data for inference.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the inference results.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference."
    )
    parser.add_argument(
        "--ivt_vars",
        nargs="+",
        default=["ivt", "ivtu", "ivtv"],
        help="Name of the IVT variables in the dataset (default: ['ivt', 'ivtu', 'ivtv'])",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        help="Type of DeepAR model to use (default: 'vit_b'). Options include 'vit_b', 'vit_l'.",
    )

    return parser.parse_args()


@torch.inference_mode()
def run_inference(model, dataloader, accelerator, progress_bar: Optional[tqdm] = None):
    all_predictions = []
    all_timestamps = []

    if progress_bar is None:
        iterator = tqdm(
            dataloader,
            desc="Running Inference",
            disable=not accelerator.is_main_process,
        )
    else:
        iterator = dataloader

    for batch in iterator:
        images = batch["image"]
        metadata = batch.get("metadata", {})

        predictions = model(images)

        output_mask = torch.sigmoid(
            predictions["output"].squeeze(1).contiguous()
        )  # Squeeze channel dim: output shape [B, 1, H, W] -> [B, H, W]

        # Gather the tensors from all GPUs to the CPU
        gathered_mask = accelerator.gather_for_metrics(output_mask).cpu().numpy()
        all_predictions.append(gathered_mask)

        if "timestamp" in metadata:
            from accelerate.utils import gather_object

            gathered_ts = gather_object(metadata["timestamp"])
            all_timestamps.extend(gathered_ts)

    all_predictions = np.concatenate(all_predictions, axis=0)

    return all_predictions, all_timestamps


def main():
    args = parse_args()
    accelerator = Accelerator()
    # Load the model
    if args.model_path is None:
        raise ValueError("Path to model weights must be provided for inference.")
    model = deep_ar_model_registry[args.model_type](checkpoint=args.model_path)

    model.eval()

    # Load the data using the inbuilt dataset class in DeepAR
    # ARInferenceDataset expects a list of files. If args.data_path is a directory, collect all .nc files.
    if os.path.isdir(args.data_path):
        input_files = [
            os.path.join(args.data_path, f)
            for f in os.listdir(args.data_path)
            if f.endswith(".nc")
        ]
    else:
        input_files = [args.data_path]

    model = accelerator.prepare(model)

    # If there are multiple files, output_path must be treated as a directory
    if len(input_files) > 1 or os.path.isdir(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
        is_output_dir = True
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        is_output_dir = False

    global_progress_bar = tqdm(
        total=len(input_files),
        desc="Inference Progress",
        disable=not accelerator.is_main_process,
    )

    for file_path in input_files:
        if accelerator.is_main_process:
            global_progress_bar.set_postfix(
                {"Current File": os.path.basename(file_path)}, refresh=False
            )

        dataset = ARInferenceDataset([file_path], ivt_vars=args.ivt_vars)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        dataloader = accelerator.prepare(dataloader)

        # Run Inference
        predictions, timestamps = run_inference(
            model, dataloader, accelerator, progress_bar=False
        )

        # Truncate to the exact dataset length (gather_object might grab padding)
        predictions = predictions[: len(dataset)]
        if len(timestamps) > 0:
            timestamps = timestamps[: len(dataset)]

        if accelerator.is_main_process:
            # Output file routing
            if is_output_dir:
                out_name = os.path.basename(file_path).replace(".nc", "_inference.nc")
                out_path = os.path.join(args.output_path, out_name)
            else:
                out_path = args.output_path

            # Get original coords
            ds_orig = xr.open_dataset(file_path)
            lats = (
                ds_orig.latitude.values if "latitude" in ds_orig else ds_orig.lat.values
            )
            lons = (
                ds_orig.longitude.values
                if "longitude" in ds_orig
                else ds_orig.lon.values
            )

            # Determine time index
            if len(timestamps) > 0:
                if isinstance(timestamps[0], str):
                    time_index = pd.to_datetime(timestamps)
                else:
                    time_index = timestamps
            else:
                # Fallback if timestamps empty or not returned
                time_index = range(predictions.shape[0])

            ds_out = xr.Dataset(
                data_vars={"ar_mask": (("time", "lat", "lon"), predictions)},
                coords={"time": time_index, "lat": lats, "lon": lons},
            )

            ds_out.to_netcdf(out_path)
            ds_orig.close()
            ds_out.close()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            global_progress_bar.update(1)

    if accelerator.is_main_process:
        global_progress_bar.close()
        print("All inferences complete.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
