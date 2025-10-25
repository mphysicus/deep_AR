import xarray as xr
import numpy as np
import os
import argparse
import multiprocessing

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

        ds['ivt'] = ivt/g
        ds['ivt_u'] = ivt_u / g
        ds['ivt_v'] = ivt_v / g
        ds = ds.drop_vars(['q', 'u', 'v', 'pressure_level', 'expver', 'number'])
        ds = ds.rename({'valid_time': 'time'})
        
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.nc', '_IVT.nc'))
        ds.to_netcdf(output_file)
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def batch_process(input_dir, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    nc_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nc')]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(calculate_ivt, [(nc_file, output_dir) for nc_file in nc_files])
    
    print("Batch processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IVT from NetCDF files using multiprocessing.")
    parser.add_argument("--input_dir", type=str, help="Path to input directory containing .nc files")
    parser.add_argument("--output_dir", type=str, help="Path to output directory for processed .nc files")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers (default: all available cores)")
    
    args = parser.parse_args()
    batch_process(args.input_dir, args.output_dir, args.workers)