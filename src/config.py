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
PROC_DATA = ROOT / config["proc_data"]
# expose the gap factor from the config file
GAP_FACTOR = config["gap_factor"]
# verify that the gap factor > 1
if not isinstance(GAP_FACTOR, (int, float)) or GAP_FACTOR <= 1:
    raise ValueError(f"Gap factor must be a number greater than 1, but got {GAP_FACTOR}. Please check the config.yaml file.")

# for standstill detection
MOTION_THRESHOLD_RAD_S = config["motion_threshold_rad_s"]
STANDSTILL_MIN_S = config["standstill_min_s"]
WHEEL_RADIUS = config["wheel_radius"]
PIT_MIN_S =  config["pit_min_s"]
GRID_RATE_HZ = config["grid_rate_hz"]
CUTOFFS = config["cutoffs"]
OFFSET_CHANNELS = config["offset_channels"]
FILES = config["files"]
DISTANCE_GRID_STEP_M = config["distance_grid_step_m"]

# segmentation

K_STRAIGHT = config["k_straight"]
MIN_STRAIGHT_M =  config["min_straight_m"]   
K_CORNER = config["k_corner"]
CURV_SMOOTH_WINDOW_M = config["curvature_smooth_window_m"]
K_LINK = config["k_link"]
MAX_LINK_M = config["max_link_m"]
PHASE_SMOOTH_M = config["phase_smooth_m"]
THROTTLE_OFF = config["throttle_off"]
THROTTLE_FULL = config["throttle_full"]
BRAKE_OFF_BAR = config["brake_off_bar"]
STEER_ZERO_DEG = config["steer_zero_deg"]
DEADBAND_BRAKE_BAR_PER_M = config["deadband_brake_bar_per_m"]
DEADBAND_THROTTLE_PER_M = config["deadband_throttle_per_m"]
DEADBAND_STEER_DEG_PER_M = config["deadband_steer_deg_per_m"]


NEEDED_CHANNELS =  config["needed_channels"]
SEGMENT_FILES = config["segment_files"]
EXCLUDED_LAPS = config["excluded_laps"]
WARMUP_LAPS = config["warmup_laps"]

DELTA_SMOOTH_M = config["delta_smooth_m"]
DELTA_COLOR_CLIP_Q = config["delta_color_clip_q"]

PHASE_MIN_RUN_M = config["phase_min_run_m"]
PHASE_MAX_GAP_M= config["phase_max_gap_m"]
PHASES = config["phases"]

## corner study
CORNER_GROUPS = config["corner_groups"]

## app
SESSION_LABEL = config["session_label"]
MANUAL_EXCLUSIONS = config["manual_exclusions"]
ISSUES_FILE = ROOT / config["issues_file"]
CONTENT_DIR = ROOT / config["content_dir"]
