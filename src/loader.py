import pandas as pd
import json
from src.session import find_files, get_t0
from src.config import RAW_DATA

# this will handle the csv files
def load(name: str):
    """
    Load data from a file by name.
    input: the short name of the file.
    output: dataframe with t_s (seconds since t0)
    
    """
    
    files = find_files()
    if name not in files:
        raise ValueError(f"File {name} not found in {sorted(files)}.")
    path = files[name]
    data = pd.read_csv(path)
    t_s = (data["_timestamp"] - get_t0()) / 1e6  # convert to seconds
    data.insert(0, "t_s", t_s)
    
    return data


# for centerline.json file
def load_centerline(name: str = "centerline.json"):
    """
    load centerline data
    input: filename.json
    output: dataframe and the ength is in the attributes of the dataframe
    
    """
    # look for centerline file in raw data root directory
    centerline_path = RAW_DATA / name
    if not centerline_path.exists():
        raise FileNotFoundError(f"Centerline file {name} not found in {RAW_DATA}.")

    # check if all arrays have the same length
    with open(centerline_path, "r") as f:
        centerline = json.load(f)
    keys = ["s", "x", "y", "theta", "curvature"]
    cent_arrays = {key: centerline[key] for key in keys}
    if not(len(set([len(value) for value in cent_arrays.values()])) == 1):
        raise ValueError(f"Centerline file {name} has inconsistent lengths in its values.")
    
    # check if data is valid
    if not centerline["valid"]:
        raise ValueError(f"Centerline file {name} is not valid.")
    
    df = pd.DataFrame({k: centerline[k] for k in keys})
    df.attrs["length"] = centerline["length"]
    
    return df
    
    
    
print(load_centerline("centerline.json"))
print(load_centerline("centerline.json").attrs["length"])