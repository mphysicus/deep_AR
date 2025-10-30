import xarray as xr
from tqdm import tqdm
import argparse
from pathlib import Path

# Load the dataset
def align_climate_grid(
        input_dir: Path,
        output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(list(input_dir.glob("*.nc")), desc="Aligning climate grid"):
        ds = xr.open_dataset(file_path)

        ds_subset = ds.sel(latitude=slice(89.75, -89.75, 2),
                        longitude=slice(0, 359.5, 2))
        
        ds_subset.coords['longitude'] = (ds_subset.coords['longitude'] + 180) % 360 - 180
        ds_subset = ds_subset.sortby(ds_subset['longitude'])
        ds_subset = ds_subset.sortby(ds_subset['latitude'], ascending=True)

        output_file = output_dir / file_path.name
        ds_subset.to_netcdf(output_file)
        ds.close()
        ds_subset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align climate data grid.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .nc files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save aligned .nc files.")
    args = parser.parse_args()

    align_climate_grid(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )