import pandas as pd
import numpy as np
from src.config import (DISTANCE_GRID_STEP_M, EXCLUDED_LAPS, WARMUP_LAPS,
                        DELTA_SMOOTH_M, PHASE_MIN_RUN_M, PHASE_MAX_GAP_M)




def comparable_laps(laps: pd.DataFrame):
    """
    
    removed the excluded laps, flags warmup lap and sorts the laps from fastest to slowest
    
    """
    ok = laps[~laps["has_pit"] & ~laps["lap"].isin(EXCLUDED_LAPS)]
    ok = ok.sort_values("lap_time").reset_index(drop=True)
    ok["is_warmup"] = ok["lap"].isin(WARMUP_LAPS)
    return ok

def reference_lap(laps: pd.DataFrame):
    """
    
    takes the fastest comparable lap of the whole session as teh reference lap for the comparison
    
    """
    return int(comparable_laps(laps).iloc[0]["lap"])


def time_delta(grid: pd.DataFrame, ref: int, lap: int,
               smooth_m: float = DELTA_SMOOTH_M):
    """
    time delta of lap vs ref along the distance grid
    delta: cumulative time lost (+) / gained (-) since the start line s
    loss_rate: how fast delta grows at each point, ms per metre,
                averaged over smooth_m metres (the raw slope is jittery)
    """
    g = grid[grid["lap"].isin([ref, lap])].copy()
    g["s"] = g["s"].round(3) # identical grid points across laps so tha they match
    # turns to long lap rows table to one column per lap, s is index
    t = g.pivot(index="s", columns="lap", values="t_lap") 

    out = pd.DataFrame({"s": t.index.to_numpy(), "delta": (t[lap] - t[ref]).to_numpy()})
    # 10 / 0.3 ~ 33 points
    window = max(1, round(smooth_m / DISTANCE_GRID_STEP_M))
    # subtract the ref from lap
    slope = out["delta"].diff() / DISTANCE_GRID_STEP_M * 1000   # ms lost per metre (ms/s)
    #  avg over 10m bc the point-to-point slope is jittery (33 grid points = 10m)
    out["loss_rate"] = slope.rolling(window, center=True, min_periods=1).mean() # calculate loss and gain of time
    return out


# finding corner windows
def _runs(mask: np.ndarray, n_min: int, gap_max: int) -> list[tuple[int, int]]:
    """
    (start, end) index pairs of runs of True; end is exclusive.
    runs separated by fewer than gap_max False points are merged into one
    merged runs shorter than n_min points are dropped.
    """
    raw, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            raw.append([start, i])
            start = None

    merged = []
    for r in raw:
        if merged and r[0] - merged[-1][1] < gap_max:
            merged[-1][1] = r[1]                   # bridge the short interruption
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= n_min]



def corner_entries(grid: pd.DataFrame, apexes: pd.DataFrame, zones: pd.DataFrame,
                   laps_used: list[int], length: float,
                   min_run_m: float = PHASE_MIN_RUN_M,
                   max_gap_m: float = PHASE_MAX_GAP_M):
    """
    per (lap, corner): where the corner entry starts, searched between the previous corner's
    last apex and this corner's first apex.
      1. braking: start of the longest braking run
      2. else trail braking: start of the longest trail braking run
      3. else lift: start of the lift run closest to this corner's apex
      4. else the corner zone start
    runs already on at the previous apex (they belong to the previous corner) are dropped
    entry_source says which rule was used.
    """
    n_min = max(1, int(np.ceil(min_run_m / DISTANCE_GRID_STEP_M)))   # metres → grid points
    gap_max = max(1, int(np.ceil(max_gap_m / DISTANCE_GRID_STEP_M)))
    cz = zones[zones["type"] == "corner"].sort_values("zone").set_index("corner")
    corners = cz.index.tolist()  # driving order
    main_apex = apexes[apexes["is_main"]].set_index("corner")["s"]

    rules = [("braking", "longest"), ("trail braking", "longest"), ("lift", "closest")]
    rows = []
    for lap, lg in grid[grid["lap"].isin(laps_used)].groupby("lap"):
        s = lg["s"].to_numpy()
        phase = lg["phase"].to_numpy()
        for i, c in enumerate(corners):
            a, p = main_apex[c], main_apex[corners[i - 1]]  # i - 1 = -1 wraps to the last corner
            d = np.mod(s - p, length) # metres after the previous main apex
            idx = np.where((d > 0) & (d < np.mod(a - p, length)))[0]
            idx = idx[np.argsort(d[idx])] # driving order
            ph = phase[idx]

            s_entry, source = float(cz.loc[c, "s_start"]), "zone"     # rule 4 by default
            for name, pick in rules:
                runs = [r for r in _runs(ph == name, n_min, gap_max) if r[0] > 0]
                if runs:
                    r = max(runs, key=lambda r: r[1] - r[0]) if pick == "longest" else runs[-1]
                    s_entry, source = float(s[idx[r[0]]]), name
                    break
            rows.append({"lap": int(lap), "corner": int(c), "s_entry": s_entry,
                         "entry_source": source})
    return pd.DataFrame(rows)


def _in_nan_run(e: float, runs: list[tuple[float, float]], length: float):
    """
    
    True if position e lies inside any NaN run of s
    
    """
    return any(np.mod(e - a, length) <= np.mod(b - a, length) for a, b in runs)


def _t_at(lg: pd.DataFrame, s: float):
    """
    t_lap at distance s
    lg = one lap, sorted by s
    
    """
    return float(np.interp(s, lg["s"], lg["t_lap"]))



def _window_time(lg: pd.DataFrame, s0: float, s1: float):
    """
    
    time spent between s0 and s1
    if the window crosses the line then the two parts are added
    
    """
    if s1 >= s0:
        return _t_at(lg, s1) - _t_at(lg, s0)
    return (float(lg["t_lap"].iloc[-1]) - _t_at(lg, s0)) + _t_at(lg, s1)


def corner_times(grid: pd.DataFrame, windows: pd.DataFrame, gaps: pd.DataFrame,
                 laps_used: list[int], ref: int, length: float):
    """
    one row per (lap, corner): time in the window, reference time, loss (+ = slower),
    valid = neither edge inside a NaN run of s, in this lap AND in the reference lap
    """
    g_ok = gaps.dropna(subset=["s_start", "s_end"])
    runs = {lap: list(zip(gl["s_start"], gl["s_end"])) for lap, gl in g_ok.groupby("lap")}

    rows = []
    for lap, lg in grid[grid["lap"].isin(laps_used)].groupby("lap"):
        lg = lg.sort_values("s")
        for _, w in windows.iterrows():
            edges_ok = not any(_in_nan_run(e, runs.get(lap, []), length)
                               for e in (w["s_start"], w["s_end"]))
            rows.append({"lap": int(lap), "corner": int(w["corner"]),
                         "time": _window_time(lg, w["s_start"], w["s_end"]),
                         "edges_ok": edges_ok})

    out = pd.DataFrame(rows)
    r = out[out["lap"] == ref].set_index("corner")
    out["ref_time"] = out["corner"].map(r["time"])
    out["valid"] = out["edges_ok"] & out["corner"].map(r["edges_ok"])
    out["loss"] = (out["time"] - out["ref_time"]).where(out["valid"])
    return out








