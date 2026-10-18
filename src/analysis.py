import pandas as pd
from src.config import EXCLUDED_LAPS, WARMUP_LAPS, DISTANCE_GRID_STEP_M




def comparable_laps(laps):
    """
    
    removed the excluded laps, flags warmup lap and sorts the laps from fastest to slowest
    
    """
    ok = laps[~laps["has_pit"] & ~laps["lap"].isin(EXCLUDED_LAPS)]
    ok = ok.sort_values("lap_time").reset_index(drop=True)
    ok["is_warmup"] = ok["lap"].isin(WARMUP_LAPS)
    return ok

def reference_lap(laps):
    """
    
    takes the fastest comparable lap of the whole session as teh reference lap for the comparison
    
    """
    return int(comparable_laps(laps).iloc[0]["lap"])


def time_delta(grid, ref, lap, smooth_m=10):
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


