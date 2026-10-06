"""
Project configuration file. This file contains the paths to the raw data, cache file, output figures, and output logs.
The paths are defined in a YAML file located at the root of the project. 

The config.py file reads the YAML file and defines the paths as constants that can be used throughout the project.
it also exposes settings such as the grap factor
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

# expose the gap factor from the config file
GAP_FACTOR = config["gap_factor"]
# verify that the gap factor > 1
if not isinstance(GAP_FACTOR, (int, float)) or GAP_FACTOR <= 1:
    raise ValueError(f"Gap factor must be a number greater than 1, but got {GAP_FACTOR}. Please check the config.yaml file.")

# for standstill detection
MOTION_THRESHOLD_RAD_S = config["motion_threshold_rad_s"]
STANDSTILL_MIN_S = config["standstill_min_s"]

