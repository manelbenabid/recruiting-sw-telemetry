
from src.config import PIT_MIN_S, PROC_DATA, MOTION_THRESHOLD_RAD_S, STANDSTILL_MIN_S
import pandas as pd
from src.preprocess import load_processed
import numpy as np

S_COL = "vehicle_curvilinear_coordinates__s"


def find_segment_standstills(df: pd.DataFrame, threshold=MOTION_THRESHOLD_RAD_S, min_s=STANDSTILL_MIN_S):
    cleaned = df[(~df["front_angular_velocity__fl"].isna()) & (~df["front_angular_velocity__fr"].isna())]
    stopped = (cleaned["front_angular_velocity__fl"].abs() < threshold) & (cleaned["front_angular_velocity__fr"].abs() < threshold)

    run_id = stopped.ne(stopped.shift()).cumsum()

    runs = cleaned.groupby(run_id).agg(start_s=("t_s", "min"), end_s=("t_s", "max"))
    runs["stopped"] = stopped.groupby(run_id).first()
    runs["duration_s"] = runs["end_s"] - runs["start_s"]

    # keep stopped runs that last long enough
    runs = runs[runs["stopped"] & (runs["duration_s"] >= min_s)]

    return runs[["start_s", "end_s", "duration_s"]].reset_index(drop=True)


def classify_standstills(df: pd.DataFrame, pit_min_s: int = PIT_MIN_S):
    standstills = find_segment_standstills(df)
    standstills["type"] = np.where(standstills["duration_s"] >= pit_min_s, "pit", "stop")

    return standstills


    


def define_stints(df: pd.DataFrame, track_length, mition_threashold: int = MOTION_THRESHOLD_RAD_S, pit_min_s: int = PIT_MIN_S, standstill_min_s = STANDSTILL_MIN_S):
    standstills = classify_standstills(df, pit_min_s)
    pits = standstills[standstills["type"] == "pit"]
    t = df["t_s"]

    starts = [t.iloc[0]] + pits["end_s"].tolist()
    ends = pits["start_s"].tolist() + [t.iloc[-1]]

    windows = pd.DataFrame({
        "start_s": starts,
        "end_s": ends
    })

    # remove windows with zero or negative length
    windows = windows[windows["end_s"] > windows["start_s"]]


    # a stint cant be < lap length
    def distance(row):
        s = df.loc[(t >= row.start_s) & (t <= row.end_s), S_COL].dropna().to_numpy()
        if len(s) < 2:
            return 0.0
        s_unwrapped = np.unwrap(s, period=track_length)
        return s_unwrapped.max() - s_unwrapped.min()

    windows["distance_m"] = windows.apply(distance, axis=1)

    
    stints = windows[windows["distance_s"] >= track_length].reset_index(drop=True)
    stints.insert(0, "stint", np.arange(1, len(stints) + 1))

    # label the rows
    df = df.copy()
    df["stint"] = np.nan
    for row in stints.itertuples():
        df.loc[(t >= row.start_s) & (t <= row.end_s), "stint"] = row.stint

    return df, stints

def unwrap(df: pd.DataFrame, col: str, period: float):
    
    """
    
    period is the full period of that column
    for exampe distance is 0 -> 805 so period = 805
    
    """
    
    data = df.copy() 
    s = df[col].dropna().to_numpy()
    data["s_unwrapped"] = pd.Series(np.unwrap(s, period=period), index=s.index)
    
    return data



    
