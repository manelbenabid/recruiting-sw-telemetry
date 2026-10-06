"""
this file aims to derive information I can't see because of the huge amount of data.
to flag any missing data, duplicates, invalid data, gaps in recordings, etc.

two files are generated in the log directory, one row per file and one row per signal column
"""
from src.config import GAP_FACTOR, OUT_LOGS
import pandas as pd
from src.session import find_files
from src.loader import load

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

# finding gaps
def find_gaps(df: pd.DataFrame, gap_factor = GAP_FACTOR):
    """
    find the gaps in one loaded data file.
     A gap is a step between consecutive samples of at least gap_factor *
    the file's median step T. Steps are taken on 't_s'.
    
    input: 
    df : DataFrame returned by loader.load() with a 't_s' column.
    gap_factor : threshold multiple of T. Defaults to the value in config.yaml.

    output:
    dataFrame with one row per gap:
        start_s    : time of the last sample before the gap (s)
        end_s      : time of the first sample after the gap (s)
        duration_s : end_s - start_s (s)
        lost_s     : duration_s - T, the time actually missing (s)
    Empty if the file has no gaps.

    raises ValueError if gap_factor is not a number greater than 1.
    
    """
    
    # ensure the gap factor is greater than 1
    if not isinstance(gap_factor, (int, float)) or gap_factor <= 1:
        raise ValueError("gap_factor must be a number greater than 1")    
    
    # steps
    dt = df['t_s'].diff().dropna()
    T  = dt[dt>0].median() # median time between samples
    # gaps, i.e. the steps where dt >= gap_factor * T
    gaps = dt[dt >= gap_factor * T]
    duration_s = gaps
    end_s = df.loc[gaps.index, "t_s"]
    start_s = end_s - gaps
    lost_s = gaps - T
     
    data = pd.DataFrame(
        {
            'start_s': start_s,
            'end_s': end_s,
            'duration_s': duration_s,
            'lost_s': lost_s
        }
    ).reset_index(drop=True)
    
    return data

def profile_file(df: pd.DataFrame, file_name: str, gap_factor = GAP_FACTOR):
    """
    profile the timing of one loaded data file.
    
    input: df: dataframe returned by loader.load() with 't_s' and '_timestamp'
    gap_factor: a step between consecutive samples cunts as a gap when it it's at least gap_factor * the median step.
    
    output:
    dict with start, end and duration (s), sample count, whether the timestamps are sorted, number of duplicate timestapms, sampling rate (Hz), number of gaps,
    largest step (s), and total time lost in gaps (s)   
    
    """
    
    
    # ensure the gap factor is greater than 1
    #if not isinstance(gap_factor, (int, float)) or gap_factor <= 1:
    #    raise ValueError("gap_factor must be a number greater than 1")    
    start_s = df['t_s'].min()
    end_s = df['t_s'].max()
    duration_s = end_s - start_s
    n_samples = len(df)
    
    # steps
    dt = df['t_s'].diff().dropna() # the dropna is for the first row
    # are the timestamps sorted?
    is_sorted = df['t_s'].is_monotonic_increasing
    # check for duplicates
    n_duplicates = df["_timestamp"].duplicated().sum()
    # rate
    T  = dt[dt>0].median() # median time between samples
    rate_hz = 1/T if T>0 else float('nan')
    # gaps, i.e. the steps where dt >= gap_factor * T
    gaps = find_gaps(df, gap_factor)
    max_gap_s = dt.max()
    n_gaps = len(gaps)
    gap_time_s = gaps["lost_s"].sum() # sum of gaps in the sample
    
    return {
        "file": file_name,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "n_samples": n_samples,
        "is_sorted": is_sorted,
        "n_duplicates": n_duplicates,
        "rate_hz": rate_hz,
        "n_gaps": n_gaps,
        "max_gap_s": max_gap_s,
        "gap_time_s": gap_time_s,
    }

# investigating column stats
def profile_columns(df: pd.DataFrame, file_name: str):
    
    """
    profile the values of each signal column in a file.
    
    param: df from loader.load() and the short_name
    
    output: 
    list of dicts, one per signal column.
    with the number of missing values, min, max, number of unique values,
    and the longest run of identical consecutive values: its length in
    samples, its duration (s) and its start time (s).
    
    """
    
    
    rows = []
    for col in df.columns:
        if col in ("t_s", "_timestamp"):
            continue
        
        s = df[col]
        # each change of value starts a new run; cumsum gives every run its own id
        run_id = s.ne(s.shift()).cumsum()
        run_sizes = s.groupby(run_id).size()
        longest_id = run_sizes.idxmax()
        run_t = df.loc[run_id == longest_id, "t_s"]

        rows.append({
            "file": file_name,
            "column": col,
            "n_nan": s.isna().sum(),
            "min": s.min(),
            "max": s.max(),
            "n_unique": s.nunique(),
            "longest_run": run_sizes.max(),
            "longest_run_s": run_t.max() - run_t.min(),
            "longest_run_start_s": run_t.min(),
        })

    return rows
    

## the final boss
def run_profile():
    
    """
    run the profile_file and profile_columns analysis on all files and save them into the logs.
    
    """
    
    
    file_rows = []
    column_rows = []
    files = find_files()
    for file in files.keys():
        df = load(file)
        file_rows.append(profile_file(df, file_name = file))
        column_rows.extend(profile_columns(df, file))
        
    
    file_rows_df = pd.DataFrame(file_rows)
    column_rows_df = pd.DataFrame(column_rows)
    
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    
    #write to output files
    file_rows_df.to_csv(OUT_LOGS / 'profile_files.csv', index=False)
    column_rows_df.to_csv(OUT_LOGS / 'profile_columns.csv', index=False)
    
    
    return file_rows_df,column_rows_df



# finding common gaps between multiple files


## find T 
def median_step(df: pd.DataFrame):
    """
    output:
    T, the median time between consecutive samples of a loaded file (s).
    """
    dt = df["t_s"].diff().dropna()
    return dt[dt > 0].median()


# find the gaps
def find_common_gaps(gap_factor=GAP_FACTOR):
    """
    Find the gaps shared by all raw data files.

    only gaps long enough to be detectable in every file are compared
    cutoff = gap_factor x the largest median step among the files, since the
    slowest file cannot register shorter gaps. 
    the slowest file is used as the reference. 
    a reference gap is common if every file has a gap overlapping it
    its interval is the union of all overlapping gaps.

    output
    dataFrame with one row per common gap: start_s, end_s, duration_s.
    Empty if there are none.
    """
    files = find_files()

    steps = {}
    all_gaps = []
    for name in files:
        df = load(name)
        steps[name] = median_step(df)
        g = find_gaps(df, gap_factor)
        g["file"] = name
        all_gaps.append(g)
    all_gaps = pd.concat(all_gaps, ignore_index=True)

    # keep only gaps every file could detect
    cutoff = gap_factor * max(steps.values())
    all_gaps = all_gaps[all_gaps["duration_s"] >= cutoff]

    # name of the file with the largest T
    ref = max(steps, key=steps.get)

    rows = []
    for _, r in all_gaps[all_gaps["file"] == ref].iterrows():
        # every kept gap overlapping this reference gap
        overlap = all_gaps[
            (all_gaps["start_s"] < r["end_s"]) & (r["start_s"] < all_gaps["end_s"])
        ]
        # common only if every file has an overlapping gap
        if set(overlap["file"]) == set(files):
            start = overlap["start_s"].min()
            end = overlap["end_s"].max()
            rows.append({"start_s": start, "end_s": end, "duration_s": end - start})

    return pd.DataFrame(rows, columns=["start_s", "end_s", "duration_s"])



from src.config import MOTION_THRESHOLD_RAD_S, STANDSTILL_MIN_S

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