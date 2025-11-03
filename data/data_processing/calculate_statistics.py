import numpy as np
import xarray as xr
from pathlib import Path
import json
from tqdm import tqdm
import argparse

def compute_statistics(data_dir, output_file):
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.nc"))

    print(f"Found {len(files)} files in {data_dir}")

    stats = {}

    for file in tqdm(files, desc="Computing statistics"):
        ds = xr.open_dataset(file)
        for var in ds.data_vars:
            if var not in stats:
                stats[var] = {'sum': 0, 'sum_sq': 0, 'count': 0}
            
            data = ds[var].values
            mask = ~np.isnan(data)
            valid_data = data[mask]

            stats[var]['sum'] += valid_data.sum()
            stats[var]['sum_sq'] += (valid_data ** 2).sum()
            stats[var]['count'] += len(valid_data)
        ds.close()

    results = {}
    for var in stats:
        mean = stats[var]['sum'] / stats[var]['count'] if stats[var]['count'] > 0 else 0
        std = np.sqrt(stats[var]['sum_sq'] / stats[var]['count'] - mean ** 2) if stats[var]['count'] > 0 else 0

        results[var] = {
            'mean': float(mean),
            'std': float(std)
        }
        print(f"Variable: {var}, Mean: {mean}, Std: {std}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Statistics saved to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean and std dev for variables in NetCDF files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing NetCDF files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file to save statistics.")
    
    args = parser.parse_args()
    
    compute_statistics(args.data_dir, args.output_file)