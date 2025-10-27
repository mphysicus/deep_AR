"""
Merge monthly IVT files into yearly files for the purposes of data loading efficiency during training.
"""

import xarray as xr
from pathlib import Path
from tqdm import tqdm
import argparse

def merge_monthly_to_yearly(
        input_dir: Path, 
        output_dir: Path,
        start_year: int,
        end_year: int
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in tqdm(range(start_year, end_year+1), desc="Merging monthly files to yearly"):
        monthly_files = []

        #Collect all 12 monthly files for this year
        for month in range(1, 13):
            file_path = input_dir / f"{year}_{month:02d}_IVT.nc"
            if file_path.exists():
                monthly_files.append(file_path)
            else:
                print(f"Warning: Missing {file_path}")
        
        if len(monthly_files) == 0:
            print(f"No files found for year {year}, skipping.")
            continue

        # Open all monthly files
        datasets = [xr.open_dataset(f) for f in monthly_files]

        # Concatenate along the time dimension
        merged = xr.concat(datasets, dim="time")

        # Sort by time (ensure chronological order)
        merged = merged.sortby("time")

        # Save merged yearly file
        output_file = output_dir / f"IVT_yearly_{year}.nc"
        merged.to_netcdf(output_file)

        # Close datasets
        for ds in datasets:
            ds.close()
        merged.close()

        print(f"Created {output_file} with {len(merged.time)} time steps.")

parser = argparse.ArgumentParser(description="Merge monthly IVT files into yearly files.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing monthly IVT .nc files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save yearly IVT .nc files.")
parser.add_argument("--start_year", type=int, required=True, help="Start year for merging.")
parser.add_argument("--end_year", type=int, required=True, help="End year for merging.")
args = parser.parse_args()

if __name__ == "__main__":
    merge_monthly_to_yearly(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        start_year=args.start_year,
        end_year=args.end_year
    )