import requests
from tqdm import tqdm

def pikart_data_download(year, catalog_version='eulerian'):
    """
    Download a single year's PIKART NetCDF file
    from the THREDDS Data Server using HTTPServer access,
    with a visible progress bar.
    """
    # Construct the HTTPServer URL for the dataset
    http_url = (
        f"https://ar.pik-potsdam.de/thredds/fileServer/"
        f"{catalog_version}_era5/PIKARTV1_{catalog_version}_ERA5_0p5deg_6hr_{year}.nc"
    )
    save_path = f"/storage/prakhar/Sem_Project/ARISe/data/data_download/pikart/PIKARTV1_{catalog_version}_ERA5_0p5deg_6hr_{year}.nc"

    print(f"Downloading {http_url}")
    response = requests.get(http_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 * 1024  # 1 MB

    with open(save_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=save_path,
        unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Full dataset saved to: {save_path}")

for year in range(1940, 2024):
    pikart_data_download(year=year)