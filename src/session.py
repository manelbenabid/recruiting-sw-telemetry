from src.config import RAW_DATA
import pandas as pd


# search recursively for files and return the short name to path mapping
def find_files(root_path=RAW_DATA, delimiter="."):
    """
    Search recursively for files in the given root path and return a dictionary mapping the short name of each file to its full path.

    """
    files = {}
    for file in root_path.rglob("*"):
        if not file.is_file():
            continue
        ## exclude the centerline.json file
        if not (file.name.endswith(".csv") or file.name.endswith(".csv.gz")):
            continue
        short_name = file.name[:file.name.find(delimiter)]
        
        if short_name in files:
            raise ValueError(f"Duplicate file name {short_name} found in {file} and {files[short_name]}.")
        
        files[short_name] =  file
        
    return files


    
# compute t0
def compute_t0():
    """
    Compute the earliest timestamp across all csv files in the raw data directory.
    """
    files = find_files(RAW_DATA)
    t0 = None # no value until the first file is read
    for path in files.values():
        file_data = pd.read_csv(path, usecols=["_timestamp"])
        t = file_data["_timestamp"].min()
        if t0 is None or t < t0:
            t0 = t
    if t0 is None:
        raise ValueError("No timestamp found in any of the csv files.")
    return int(t0)



