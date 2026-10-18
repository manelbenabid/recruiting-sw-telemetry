import pandas as pd
import numpy as np
from src.config import (DISTANCE_GRID_STEP_M, EXCLUDED_LAPS, WARMUP_LAPS,
                        DELTA_SMOOTH_M, BRAKE_OFF_BAR, PHASE_MIN_RUN_M)
from src.segment import BRAKE_COL



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
def _last_run_start(on: np.ndarray, n_min: int) -> int | None:
    """Start index of the last run of at least n_min consecutive True values.
    A run starting at index 0 is ignored (braking carried over from the previous corner)."""
    last, run = None, 0
    for i, v in enumerate(on):
        run = run + 1 if v else 0
        start = i - n_min + 1
        if run == n_min and start > 0:
            last = start
    return last


def corner_windows(grid: pd.DataFrame, zones: pd.DataFrame, laps_used: list[int],
                   length: float, min_run_m: float = BRAKE_MIN_RUN_M):
    """
    one window per corner zone: start shifted back to the earliest brake point across laps,
    where a lap's brake point = start of its last brake run in the straight before the corner
    end = corner zone end.
    """
    z = zones.sort_values("zone").reset_index(drop=True)
    g = grid[grid["lap"].isin(laps_used)]
    n_min = max(1, int(np.ceil(min_run_m / DISTANCE_GRID_STEP_M)))   # metres → grid points

    rows = []
    for i, cz in z.iterrows():
        if cz["type"] != "corner":
            continue
        start = cz["s_start"]
        prev = z.iloc[i - 1] # i - 1 = -1 wraps to the last zone
        ext, reach = 0.0, 0.0
        if prev["type"] == "straight":
            reach = float(np.mod(start - prev["s_start"], length))   # length of that straight
            back = np.mod(start - g["s"], length)  # metres before the corner start
            region = g.assign(back=back)[(back > 0) & (back <= reach)]
            for _, lg in region.groupby("lap"):
                lg = lg.sort_values("back", ascending=False) # driving order
                on = (lg[BRAKE_COL] > BRAKE_OFF_BAR).to_numpy()
                b = _last_run_start(on, n_min)
                if b is not None:
                    ext = max(ext, float(lg["back"].iloc[b])) # earliest = furthest back
        rows.append({"corner": int(cz["corner"]), "zone": int(cz["zone"]),
                     "s_start": float(np.mod(start - ext, length)),
                     "s_end": float(cz["s_end"]), "brake_ext_m": ext, "straight_m": reach})
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








