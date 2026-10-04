"""
Project configuration file. This file contains the paths to the raw data, cache file, output figures, and output logs.
The paths are defined in a YAML file located at the root of the project. 

The config.py file reads the YAML file and defines the paths as constants that can be used throughout the project.

"""

import yaml
from pathlib import Path

# get the root path of the project
ROOT = Path(__file__).resolve().parent.parent

config_path = ROOT / "config.yaml"

# open the config file and parse it as a dictionary
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# define the other paths 
RAW_DATA = ROOT / config["raw_data"]
# check if raw data path exists.
if not RAW_DATA.exists(): raise FileNotFoundError(f"Raw data path {RAW_DATA} does not exist.")


CACHE_FILE = ROOT / config["cache_file"]
OUT_FIG = ROOT / config["out_fig"]
OUT_LOGS = ROOT / config["out_logs"]