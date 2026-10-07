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

from src.config import WHEEL_RADIUS, PIT_MIN_S
from src.profiler import find_common_gaps
from src.loader import load_centerline

def find_laps():
    
    """
    split the session into laps at start/finish line crossings.
    
    a "crossing" is where s drops by more than hald the track length 
    while the dront wheels are moving (rule1).
    crossings closeer than the minimum lap time (track length / session top speed)
    to the previous accepted one are discarded as flicker (rule2)
    
    output:
    a df with one row per lap:
    lap number | start_s | end_s | lap_time_s | has_stop | has_gap | has_pit | is_out_lap | valid
    the flags mean the following:
    has_stop: it contains a standstill
    has_gap: contains a common gap
    has_pit: contains a pit stop
    is_out_lap: boxed and is coming out of the pit
    valid: no stop, no gap, nt an outlap
    """
    
    
    
    df = load("vehicle_curvilinear_coordinates")
    standstills = find_standstills()
    common_gaps = find_common_gaps()
    
    pits = standstills[standstills["duration_s"] >= PIT_MIN_S]

    
    ds = df["s"].diff()
    track_length = load_centerline().attrs["length"]
    cand = df[ds < -track_length / 2]
    
    # rule1: wheels are moving
    wheel_speed = load("front_angular_velocity")
    survivors = pd.merge_asof(cand, wheel_speed, direction="nearest", on="t_s")
    survivors = survivors[(survivors["fl"].abs() >= MOTION_THRESHOLD_RAD_S) | (survivors["fr"].abs() >= MOTION_THRESHOLD_RAD_S)]
    
    # rule2: minimum lap time = track length / session's top speed
    speed = ((wheel_speed["fr"] + wheel_speed["fl"]) /2) * WHEEL_RADIUS
    top_speed = speed.max()
    min_lap_s = track_length / top_speed
    
    survivors = survivors.sort_values("t_s")

    accepted = []
    last_t = None  # time of the last accepted boundary
    for t in survivors["t_s"]:
        if last_t is None or t - last_t >= min_lap_s:
            accepted.append(t)
            last_t = t

    boundaries = pd.Series(accepted, name="t_s") # rach paid boundaries[i] and boundaries[i+1] is one lap
    
    rows = []

    for i in range(len(boundaries) -1):
        lap_start = boundaries[i]
        lap_end = boundaries[i+1]
        has_stop = ((standstills["start_s"] < lap_end) & (lap_start < standstills["end_s"])).any()
        has_gap  = ((common_gaps["start_s"] < lap_end) & (lap_start < common_gaps["end_s"])).any()
        has_pit = ((pits["start_s"] < lap_end) & (lap_start < pits["end_s"])).any()
        is_out_lap = ((pits["end_s"] >= lap_start) & (pits["end_s"] < lap_end)).any()
        
        valid = not (has_stop or has_gap or is_out_lap)
        rows.append({
            "lap_number": i+1,
            "start_s": lap_start,
            "end_s": lap_end,
            "lap_time_s": lap_end - lap_start,
            "has_stop": has_stop,
            "has_gap": has_gap,
            "has_pit": has_pit,
            "is_out_lap": is_out_lap,
            "valid": valid
        })
        
        

    laps = pd.DataFrame(rows)
    
    return laps
    
        
    