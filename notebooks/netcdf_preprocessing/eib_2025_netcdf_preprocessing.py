# This preprocessing script processes the netcdf weather parameter files into data cubes 
# and saves the data in numpy binary format for each timestep
# Contributed by Selorm Komla Darkey (skdarkey@gmail.com)
import pandas as pd 
import os
import numpy as np
from datetime import datetime
import rasterio as rio
from pathlib import Path


# function to get the parameter components
def read_and_extract_param_data(path):
    """
        Arguments:
            - Takes a path to the climate file as input

        Output:
            - It extracts the timesteps for each band in the file
            - Also extracts the climate parameter numeric data stored for each band in the file
            - Returns the time steps, and the data for each band in the file.
    """
    with rio.open(path) as src:
        # Read all bands
        data = src.read()

        time_stamps = []
        for i in range(src.count):
            tag_index = i + 1
            netcdf_time = src.tags(tag_index).get('NETCDF_DIM_time')
            grib_time = src.tags(tag_index).get('GRIB_REF_TIME')

            if netcdf_time is not None:
                # NETCDF_DIM_time is in hours since 1970-01-01
                try:
                    dt = datetime(1970, 1, 1) + timedelta(hours=int(netcdf_time))
                    time_stamps.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
                except ValueError:
                    time_stamps.append("Invalid NETCDF_DIM_time")
            elif grib_time is not None:
                # GRIB_REF_TIME is in seconds since 1970-01-01
                try:
                    dt = datetime.utcfromtimestamp(int(grib_time))
                    time_stamps.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
                except ValueError:
                    time_stamps.append("Invalid GRIB_REF_TIME")
            else:
                time_stamps.append("Unknown Time Format")

        return time_stamps, data


# function to load all params data into a single python dictionary
def load_all_parameter_data(paths):
    """
        Arguments:
            - Creates a dictionary of the data and timestep for each of the climate 
            parameter.
        Output:
            - Returns a data dictionary that contains Timesteps and numeric parameter values
                stored for each of the 3 climate params.

    """
    # create empty dict
    all_data = {}

    for param_name, path in paths.items():
        times, data = read_and_extract_param_data(path)
        all_data[param_name] = {
            'times': times,
            'data': data
        }

    return all_data


# function to stack the parameter data for each time step without overlap
def save_each_timestep_stack(parameter_data, save_dir='output_arrays'):
    """Arguments: 
            - Takes in a dictionary of data of the 3 climate variables prepared.
            - The input data dictionary should contain timesteps and the parameter values 
        Output:
            - A numpy data file is created to store the stacked values for the 3 climate 
                parameters at each timestep.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract and intersect common timestamps across all variables
    time_sets = [set(var['times']) for var in parameter_data.values()]
    common_times = sorted(set.intersection(*time_sets), key=lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))

    if not common_times:
        print("No common timestamps found across variables.")
        return

    # Create timestamp-to-index mapping for each variable
    time_index_map = {
        var_name: {t: i for i, t in enumerate(var_data['times'])}
        for var_name, var_data in parameter_data.items()
    }

    # Get sample shape for sanity
    sample_data = next(iter(parameter_data.values()))['data']
    _, height, width = sample_data.shape

    for t in common_times:
        stacked_layers = []

        for var_name in ['mslp', 'geo_pht', 'wbpt']:  # Ensures consistent order
            idx = time_index_map[var_name][t]
            layer = parameter_data[var_name]['data'][idx]
            stacked_layers.append(layer)

        stacked_array = np.stack(stacked_layers, axis=0)  # Shape: (3, H, W)

        # Format filename
        fname = t.replace(':', '').replace(' ', '_') + '.npy'
        save_path = os.path.join(save_dir, fname)
        np.save(save_path, stacked_array)
        print(f"Saved: {save_path} | Shape: {stacked_array.shape}")


# function that extracts file paths and dates into csv
def read_and_write_into_csv(path, output_path):
    """ 
    Argument:
        - Path: a directory path where the individual files are stored
        - Output_path: the path where the csv should be stored
    """
    files = os.listdir(path)

    all_files = []

    for filename in files:
        file_path = os.path.join(path, filename)
        date = os.path.splitext(filename)[0]  

        all_files.append({'path': file_path, 'date': date})

    df = pd.DataFrame(all_files)
    df.to_csv(output_path, index=False)


def main():
    # set paths to the 3 weather parameter files 
    paths_2024 = {
        'mslp': Path(r"C:\PERSONAL\UK PHD\2025_eib_hackerton\2024_air_pressure_at_sea_level.nc"),
        'geo_pht': Path(r"C:\PERSONAL\UK PHD\2025_eib_hackerton\2024_geopotential_height.nc"),
        'wbpt': Path(r"C:\PERSONAL\UK PHD\2025_eib_hackerton\2024_wbpt.nc"),
    }
    # extract the data for the 3 weather parameters
    parameters_data = load_all_parameter_data(paths_2024)

    # stack and separate the data into each time step and create .npy files
    output_files_dir = r"C:\PERSONAL\UK PHD\2025_eib_hackerton\npy_files_2024_single_timesteps"
    save_each_timestep_stack(parameters_data, output_files_dir)

    # write to .npy file into csv 
    npy_files_path = r"C:\PERSONAL\UK PHD\2025_eib_hackerton\npy_files_2024_single_timesteps"
    out_csv_file_path = r"C:\PERSONAL\UK PHD\2025_eib_hackerton\meteo_data_paths.csv"

    read_and_write_into_csv(npy_files_path, out_csv_file_path)


if __name__=="__main__":
    
    main()