from src.config import GAP_FACTOR, OUT_LOGS
import pandas as pd
from src.session import find_files
from src.loader import load

from src.config import MOTION_THRESHOLD_RAD_S, STANDSTILL_MIN_S

SENSORS = {
    "imu_angular_rate": ["x", "y", "z"],
    "imu_acceleration": ["x", "y", "z"],
    "front_angular_velocity": ["fl", "fr"],
    "inv_l_id_a8_n_actual_filt": ["n_act_filt"],
    "inv_r_id_a8_n_actual_filt": ["n_act_filt"],
    "inv_l_id_27_iq_actual": ["iq_act_filt"],
    "inv_r_id_27_iq_actual": ["iq_act_filt"],
    "pedal_throttle": ["throttle"],
    "pedal_brakes_pressure": ["front", "rear"],
    "hv_power": ["power"],
}

def find_standstills(threshold=MOTION_THRESHOLD_RAD_S, min_s=STANDSTILL_MIN_S):
    """
    find the periods where the car is at standstill

    the car is stopped when both front wheel speeds are below threshold (rad/s)
    and a standstill is a stopped period lasting at least min_s seconds
    both default to the values in config.yaml

    output
    a dataFrame with one row per standstill: start_s, end_s, duration_s.
    """
    df = load("front_angular_velocity")

    # both front wheels are below the threshold
    stopped = (df["fl"].abs() < threshold) & (df["fr"].abs() < threshold)

    run_id = stopped.ne(stopped.shift()).cumsum()

    runs = df.groupby(run_id).agg(start_s=("t_s", "min"), end_s=("t_s", "max"))
    runs["stopped"] = stopped.groupby(run_id).first()
    runs["duration_s"] = runs["end_s"] - runs["start_s"]

    # keep stopped runs that last long enough
    runs = runs[runs["stopped"] & (runs["duration_s"] >= min_s)]

    return runs[["start_s", "end_s", "duration_s"]].reset_index(drop=True)


# this is the function that will give me
# window | file | column | mean | std | n 
# where mean is the mean of the column in that window
# the std is the standard deviation of that column in that window
# and n is the sample count in that window

def standstill_stats(windows=None, sensors=SENSORS):
    """
    compute the offset and noise of each sensor during each standstill window.

    params
    windows : you can overrite these otherise use find_standstills()
    sensors : dict mapping a file's short name to the columns to measure

    output
    dataFrame with one row per (window, file, column):
        window: window id (row number in windows)
        start_s: window start time (s)
        file, column
        mean: mean value in the window (the offset)
        std: standard deviation in the window (the noise)
        n: number of samples in the window
    """
    if windows is None:
        windows = find_standstills()

    rows = []
    for name, columns in sensors.items():
        df = load(name) 
        for w, win in windows.iterrows():
            in_win = df[(df["t_s"] >= win["start_s"]) & (df["t_s"] <= win["end_s"])]
            for col in columns:
                rows.append({
                    "window": w,
                    "start_s": win["start_s"],
                    "file": name,
                    "column": col,
                    "mean": in_win[col].mean(),
                    "std": in_win[col].std(),
                    "n": in_win[col].count(),
                })

    return pd.DataFrame(rows)