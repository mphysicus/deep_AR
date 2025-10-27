import xarray as xr
import numpy as np
import os
import argparse
import multiprocessing
from tqdm import tqdm

def calculate_ivt(input_file, output_dir):
    try:
        ds = xr.open_dataset(input_file)

        q = ds['q']
        u = ds['u']
        v = ds['v']
        p = ds['pressure_level'] * 100  # Convert hPa to Pa
        g = 9.81  # Gravity constant
        
        dp = p.diff('pressure_level')
        dp = dp.broadcast_like(q.isel(pressure_level=slice(1, None)))
        
        ivt_u = ((q.isel(pressure_level=slice(1, None)) * u.isel(pressure_level=slice(1, None)) * dp)).sum(dim='pressure_level')
        ivt_v = ((q.isel(pressure_level=slice(1, None)) * v.isel(pressure_level=slice(1, None)) * dp)).sum(dim='pressure_level')

        ivt = np.sqrt(ivt_u**2 + ivt_v**2)

        output_ds = xr.Dataset(
            {
                'ivt': ivt / g,
                'ivt_u': ivt_u / g,
                'ivt_v': ivt_v / g,
            },
            coords=ds.coords
        ).rename({'valid_time': 'time'})
        
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.nc', '_IVT.nc'))
        output_ds.to_netcdf(output_file)
        ds.close()
        output_ds.close()
        return f"SUCCESS: {os.path.basename(output_file)}"
    except Exception as e:
        return f"ERROR: {input_file}: {e}"

def calculate_ivt_wrapper(args):
    """
    Helper function to unpack arguments for pool.imap_unordered.
    """
    return calculate_ivt(*args)

def batch_process(input_dir, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    nc_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nc')]
    
    tasks = [(nc_file, output_dir) for nc_file in nc_files]

    success_count = 0
    error_count = 0
    error_messages = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing files") as pbar:
            for result in pool.imap_unordered(calculate_ivt_wrapper, tasks, chunksize=1):
                if result.startswith("SUCCESS"):
                    success_count += 1
                elif result.startswith("ERROR"):
                    error_count += 1
                    error_messages.append(result)
                    tqdm.write(result)
                else:
                    error_count += 1
                    error_messages.append(f"UNKNOWN: {result}")
                
                pbar.update(1)
                pbar.set_postfix({'Success': success_count, 'Errors': error_count})
    
    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors: {error_count} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IVT from NetCDF files using multiprocessing.")
    parser.add_argument("--input_dir", type=str, help="Path to input directory containing .nc files")
    parser.add_argument("--output_dir", type=str, help="Path to output directory for processed .nc files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    print(f"Using {args.workers} workers for processing.")
    batch_process(args.input_dir, args.output_dir, args.workers)