import cdsapi
import time
import calendar

# Initialize the CDS API client
c = cdsapi.Client()

# Define years, pressure levels, and variables
years = list(range(1942, 2025))
pressure_levels = ["300","350", "400", "450", "500", "550", "600", "650", "700", "750", "800", "850", "900","950", "1000"]
variables = ["specific_humidity", "u_component_of_wind", "v_component_of_wind"]

# Max number of retries for failed downloads
MAX_RETRIES = 6

for year in years:
    for month in range(1, 13):
        month_str = f"{month:02d}"
        
        # Generate valid days for this month
        max_day = calendar.monthrange(year, month)[1]

        days = [f"{day:02d}" for day in range(1, max_day + 1, 5)]
        
        file_name = f"/storage/prakhar/Sem_Project/deep_AR/data/data_download/era5/{year}_{month_str}.nc"

        print(f"Requesting {file_name}...")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Make the request
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": variables,
                        "pressure_level": pressure_levels,
                        "year": str(year),
                        "month": month_str,
                        "day": days,  
                        "time": ["03:00", "09:00", "15:00", "21:00"], 
                        "download_format": "unarchived",
                        "data_format": "netcdf"
                    },
                    file_name,  
                )

                print(f"✅ Successfully downloaded: {file_name}")
                break 

            except Exception as e:
                print(f"⚠️ Attempt {attempt}/{MAX_RETRIES} failed for {file_name}: {e}")
                if attempt < MAX_RETRIES:
                    print("Retrying in 10 seconds...")
                    time.sleep(10)  # Wait before retrying
                else:
                    print(f"❌ Failed to download {file_name} after {MAX_RETRIES} attempts.")