
from src.config import PIT_MIN_S, PROC_DATA, MOTION_THRESHOLD_RAD_S, STANDSTILL_MIN_S
import pandas as pd
from src.preprocess import load_processed
import numpy as np
from src.loader import load_centerline

S_COL = "vehicle_curvilinear_coordinates__s"
FL_COL = "front_angular_velocity__fl"
FR_COL = "front_angular_velocity__fr"
U_COL = "vehicle_speed__u"

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


    


def define_stints(df: pd.DataFrame, track_length, pit_min_s: int = PIT_MIN_S):
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

    # at least one lap of distance
    stints = windows[windows["distance_m"] >= track_length].reset_index(drop=True)
    stints.insert(0, "stint", np.arange(1, len(stints) + 1))

    # each stint runs until the next one starts, so a pit belongs to the stint before it
    stints.loc[stints.index[:-1], "end_s"] = stints["start_s"].iloc[1:].to_numpy()


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

def segment_laps(df: pd.DataFrame, standstills: pd.DataFrame, track_length: float,
                 motion_threshold: float = MOTION_THRESHOLD_RAD_S):
    df = df.copy()

    # numpy arrays, worked on by position (0..n-1) so the df index doesnt matter
    t = df["t_s"].to_numpy()
    s = df[S_COL].to_numpy()
    fl = df[FL_COL].to_numpy()
    fr = df[FR_COL].to_numpy()
    stint = df["stint"].to_numpy()

    # 1. usable rows: inside a stint and s known
    usable = ~np.isnan(s) & ~np.isnan(stint)
    pos = np.flatnonzero(usable) # positions of usable rows
    s_usable, st = s[pos], stint[pos]
    moving = (np.abs(fl[pos]) > motion_threshold) & (np.abs(fr[pos]) > motion_threshold)

    # 2. candidates: s drops by more than half a lap vs the previous usablel row,
    # same stint, and the car is moving at that row
    is_candidate = np.zeros(len(pos), dtype=bool)
    is_candidate[1:] = ((s_usable[1:] - s_usable[:-1] < -track_length /2) # the diff is more than half the track length
                        & (st[1:] == st[:-1]) # same stint
                        & moving[1:]) # is moving
    cand_pos = pos[is_candidate]

    # 3. minimum lap time: keep a cand only if it comes at least
    # track_length / max speed after the last kept coundary
    min_lap_time = track_length / np.nanmax(df[U_COL].to_numpy())
    kept = []
    for p in cand_pos:
        if not kept or t[p] - t[kept[-1]] >= min_lap_time:
            kept.append(p)
    boundary_pos = np.array(kept, dtype=int) # row position (index) where each lap starts
    boundary_stint = stint[boundary_pos] 

    # 4. laps: every pair of consecutive boundaries, 
    # the lap takes the stint it starts in
    rows = []
    for k in range(len(boundary_pos) - 1):
        rows.append({"stint": int(boundary_stint[k]),
                     "start_pos": boundary_pos[k],
                     "end_pos": boundary_pos[k + 1] - 1})   # sample before next boundary
    laps = pd.DataFrame(rows, columns=["stint", "start_pos", "end_pos"])

    
    # 5. number the laps 1..N across the session
    laps.insert(0, "lap", np.arange(1, len(laps) +1))
    # 6. lap times
    laps["t_start"] = t[laps["start_pos"]]
    laps["t_end"] = t[laps["end_pos"]]
    laps["lap_time"] = laps["t_end"] - laps["t_start"]

    # 7. flags
    def overlaps(table):
        # True per lap if any window in table' overlaps [t_start, t_end]
        return [((table["start_s"] <= te) & (table["end_s"] >= ts)).any()
                for ts, te in zip(laps["t_start"], laps["t_end"])]

    laps["has_pit"] = overlaps(standstills[standstills["type"] == "pit"])
    laps["has_stop"] = overlaps(standstills[standstills["type"] == "stop"])
    laps["has_nan_s"] = [np.isnan(s[a:b + 1]).any()
                       for a, b in zip(laps["start_pos"], laps["end_pos"])]

    
    # 8. gaps table: each NaN run in n inside a lap
    gap_rows = []
    for lap, a, b in zip(laps["lap"], laps["start_pos"], laps["end_pos"]):
        nan = np.isnan(s[a:b + 1]).astype(int)
        edges = np.diff(np.concatenate([[0], nan, [0]]))
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1) - 1
        for rs, re in zip(run_starts, run_ends):
            i0, i1 = a + rs, a + re # first and last NaN position
            gap_rows.append({"lap": lap,
                             "t_start": t[i0], 
                             "t_end": t[i1],
                             "duration_s": t[i1] - t[i0],
                             "s_start": s[i0 - 1], # last valid s before the gap
                             "s_end": s[i1 + 1]})  

    gaps = pd.DataFrame(gap_rows, columns = ["lap", "t_start", "t_end", "duration_s", "s_start", "s_end"])

    # 9. label rows with their number 
    lap_col = np.full(len(df), np.nan)
    for lap, a, b in zip(laps["lap"], laps["start_pos"], laps["end_pos"]):
        lap_col[a:b+1] = lap
    df["lap"] = lap_col

    return df, laps.drop(columns=["start_pos", "end_pos"]), gaps
















    
    
