import pandas as pd
from src.session import find_files, get_t0

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