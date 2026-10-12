
from src.config import PIT_MIN_S, PROC_DATA, MOTION_THRESHOLD_RAD_S, STANDSTILL_MIN_S
import pandas as pd
from src.preprocess import load_processed
import numpy as np
from src.loader import load_centerline

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

def segment_laps(df: pd.DataFrame, standstills, stints, track_length):
    df = df.copy()

    t = df["t_s"].to_numpy()
    s = df[S_COL]
    stint = df["stint"].to_numpy()

    # 1. number of times the line has been crossed
    count = np.floor(df["s_unwrapped"].to_numpy() / track_length)
    usable = ~np.isnan(count) & ~np.isnan(stint) # inside a stint and s not NaN
    pos = np.flatnonzero(usable) # the positions of the usable rows
    c, st = count[pos], stint[pos]

    # 2. boundaries of the laps,
    is_boundary = np.zeroes(len(pos), dtype=bool)
    is_boundary[1:] = (c[1:]) & (st[1:] == st[:-1])
    b_pos = pos[is_boundary]
    b_stint = stint[b_pos]

    # 3. laps
    rows = []
    for k in range(len(b_pos) -1):
        if b_stint[k] != b_stint[k+1]:
            continue # next boundary is in another stint
        rows.append({
            "stint": int(b_stint[k]),
            "start_pos": b_pos[k],
            "end_pos": b_pos[k+1] -1
        })

    laps = pd.DataFrame(rows, columns=["stint", "start_pos", "end_pos"])

    # 4. number the laps 1..N across the session
    laps.insert(0, "lap", np.arange(1, len(laps), +1))
    # 5. lap times
    laps["t_start"] = t[laps["start_pos"]]
    laps["t_end"] = t[laps["end_pos"]]
    laps["lap_time"] = laps["t_end"] - laps["t_start"]

    # 6. flags
    stops = standstills[standstills["type"] == "stop"]
    laps["has_stop"] = [
        ((stops["start_s"] <= te) & (stops["end_s"] >= ts)).any()   # overlap test
        for ts, te in zip(laps["t_start"], laps["t_end"])
    ]

    laps["has_gap"] = [np.isnan(s[a:b + 1]).any()
                       for a, b in zip(laps["start_pos"], laps["end_pos"])]

    # line grossed inside a gap
    laps["gap_on_boundary"] = [np.isnan(s[a - 1]) or np.isnan(s[b])
                               for a, b in zip(laps["start_pos"], laps["end_pos"])]


    # 7. gaps table: each NaN run in n inside a lap
    gap_rows = []
    for lap, a, b in zip(laps["lap"], laps["start_pos"], laps["end_pos"]):
        nan = np.nan(s[a:b+1]).astype(int)
        edges = np.diff(np.concatenate([[0], nan, [0]]))
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1) - 1
        for rs, re, in zip(run_starts, run_ends):
            i0, i1=a+rs, a+re
            gap_rows.append({
                "lap": lap,
                "t_start": t[i0],
                "t_end": t[i1],
                "duration_s": t[i1] - t[i0],
                "s_start": s[i0 - 1], # last valid s before the gap
                "e_end": s[i1 +1] # first valid s after the gap
            })

    gaps = pd.DataFrame(gap_rows, columns = ["lap", "t_start", "t_end", "duration_s", "s_start", "s_end"])

    # 8. label rows with their number 
    lap_col = np.full(len(df), np.nan)
    for lap, a, b in zip(laps["lap"], laps["start_pos"], laps["end_pos"]):
        lap_col[a:b+1] = lap
    df["lap"] = lap_col

    return df, laps.drop(columns=["start_pos", "end_pos"]), gaps
















    
    
