import pandas as pd
import numpy as np
from src.config import (DISTANCE_GRID_STEP_M, EXCLUDED_LAPS, WARMUP_LAPS,
                        DELTA_SMOOTH_M, PHASE_MIN_RUN_M, PHASE_MAX_GAP_M,
                        MOTION_THRESHOLD_RAD_S, DEADBAND_STEER_DEG_PER_M,
                        THROTTLE_OFF, PHASES)

from src.segment import (BRAKE_COL, THROTTLE_COL, U_COL, FL_COL, FR_COL)
LAT_COL = "imu_acceleration__y"
RL_COL = "inv_l_id_a8_n_actual_filt__n_act_filt"
RR_COL = "inv_r_id_a8_n_actual_filt__n_act_filt"



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


def corner_times(grid: pd.DataFrame, entries: pd.DataFrame, zones: pd.DataFrame,
                 gaps: pd.DataFrame, laps_used: list[int], ref: int,
                 length: float):
    """
    one row per (lap, corner), lap vs reference
    window: from the earlier of the two laps' corner entries to the corner zone end
    entry_diff_m: how much earlier (+) or later (-) the lap enters than the reference, m.
    valid = neither window edge inside a NaN run of s, in this lap or the reference lap
    loss (+ = slower) is NaN if not valid
    """
    s_end = zones[zones["type"] == "corner"].set_index("corner")["s_end"]
    g_ok = gaps.dropna(subset=["s_start", "s_end"])
    runs = {lap: list(zip(gl["s_start"], gl["s_end"])) for lap, gl in g_ok.groupby("lap")}
    by_lap = {lap: lg.sort_values("s") for lap, lg in grid[grid["lap"].isin(laps_used)].groupby("lap")}
    ent = entries.set_index(["lap", "corner"])

    rows = []
    for lap in laps_used:
        for c, end in s_end.items():
            e_lap, e_ref = ent.loc[(lap, c)], ent.loc[(ref, c)]
            back_lap = np.mod(end - e_lap["s_entry"], length)          # metres from entry to zone end
            back_ref = np.mod(end - e_ref["s_entry"], length)
            start = e_lap["s_entry"] if back_lap >= back_ref else e_ref["s_entry"]   # the earlier one

            valid = not any(_in_nan_run(e, runs.get(l, []), length)
                            for l in (lap, ref) for e in (start, end))
            t_lap = _window_time(by_lap[lap], start, end)
            t_ref = _window_time(by_lap[ref], start, end)

            rows.append({"lap": int(lap), "corner": int(c),
                         "s_start": float(start), "s_end": float(end),
                         "entry_lap": float(e_lap["s_entry"]), "entry_ref": float(e_ref["s_entry"]),
                         "source_lap": e_lap["entry_source"], "source_ref": e_ref["entry_source"],
                         "entry_diff_m": float(back_lap - back_ref),
                         "time": t_lap, "ref_time": t_ref, "valid": valid,
                         "loss": t_lap - t_ref if valid else np.nan})
    return pd.DataFrame(rows)


## metrics


def _from_origin(s: float, origin: float, length: float):
    """
    returns the distance of a point from a reference point (origin), signed, - means before the ref, + means after the ref
    it accounts for the loop
    
    """
    return float(np.mod(s - origin + length / 2, length) - length / 2)


## this function needs function for other tracks bc it would fail if the corner spands two laps
def _stretch(lap_grid: pd.DataFrame, s0: float, s1: float, length: float):
    """
    lap_grid = rows of the grid of one lap
    returns the grid points of one lap from s0 to s1 in driving order (crosses s = 0 if needed)
    adds d = metres of grid point from s0
    
    """
    d = np.mod(lap_grid["s"].to_numpy() - s0, length)
    m = d <= np.mod(s1 - s0, length)
    return lap_grid[m].assign(d=d[m]).sort_values("d")


def corner_exits(apexes: pd.DataFrame, entries: pd.DataFrame, zones: pd.DataFrame,
                 laps_used: list[int], length: float) :
    """
    Exit point per corner, same for every lap: the corner's zone end,
    or the earliest entry of the next corner across laps if that comes first.
    Returns s_exit indexed by corner.
    """
    order = apexes[apexes["is_main"]].sort_values("s")["corner"].tolist()   # corners in driving order
    zc = zones[zones["type"] == "corner"].set_index("corner")
    e = entries[entries["lap"].isin(laps_used)]

    out = {}
    for i, c in enumerate(order):
        s_end = float(zc.loc[c, "s_end"])
        nxt = order[(i + 1) % len(order)]
        en = e[e["corner"] == nxt]["s_entry"]
        before = (np.mod(en - s_end + length / 2, length) - length / 2).min()   # − = next entry before zone end
        out[c] = float(np.mod(s_end + before, length)) if before < 0 else s_end
    return pd.Series(out, name="s_exit")

def corner_metrics(grid: pd.DataFrame, entries: pd.DataFrame, apexes: pd.DataFrame,
                   zones: pd.DataFrame, laps_used: list[int], length: float,
                   min_run_m: float = PHASE_MIN_RUN_M,
                   max_gap_m: float = PHASE_MAX_GAP_M):
    """
    one row per (lap, corner), measured from that lap's own entry to the corner's exit point
    (corner_exits: zone end, or the earliest next-corner entry across laps if that comes first)
    positions are signed metres from the corner's main apex (negative = before the apex)

    """
    n_min = max(1, int(np.ceil(min_run_m / DISTANCE_GRID_STEP_M)))
    gap_max = max(1, int(np.ceil(max_gap_m / DISTANCE_GRID_STEP_M)))
    cz = zones[zones["type"] == "corner"].sort_values("zone").set_index("corner")
    corners = cz.index.tolist()
    main_apex = apexes[apexes["is_main"]].set_index("corner")["s"]
    exits = corner_exits(apexes, entries, zones, laps_used, length)
    ent = entries.set_index(["lap", "corner"])

    rows = []
    for lap, lg in grid[grid["lap"].isin(laps_used)].groupby("lap"):
        for i, c in enumerate(corners):
            a, end = main_apex[c], exits[c]
            prev_a = main_apex[corners[i - 1]]
            e = ent.loc[(lap, c)]
            w = _stretch(lg, e["s_entry"], end, length)
            u = w[U_COL].to_numpy()
            row = {"lap": int(lap), "corner": int(c), "entry_source": e["entry_source"],
                   "entry_from_apex": _from_origin(e["s_entry"], a, length)}

            # braking, speed, throttle pickup, exit
            to_apex = w[w["d"] <= np.mod(a - e["s_entry"], length)]
            after_apex = w[w["d"] >= np.mod(a - e["s_entry"], length)]
            row["peak_brake_bar"] = float(to_apex[BRAKE_COL].max()) if len(to_apex) else np.nan

            if np.isfinite(u).any():
                k = int(np.nanargmin(u))
                row["min_speed"] = float(u[k])
                row["min_speed_from_apex"] = _from_origin(w["s"].iloc[k], a, length)
            else:
                row["min_speed"] = row["min_speed_from_apex"] = np.nan

            on = (w[THROTTLE_COL] > THROTTLE_OFF).to_numpy()
            runs = _runs(on, n_min, gap_max)
            if runs and runs[-1][1] == len(on) and runs[-1][0] > 0:
                row["pickup_from_apex"] = _from_origin(w["s"].iloc[runs[-1][0]], a, length)
            else:
                row["pickup_from_apex"] = np.nan
            row["exit_speed"] = float(u[-1]) if len(u) else np.nan

            # time and distance per phase, entry -> exit point
            t = w["t_lap"].to_numpy()
            dt = np.diff(t, append=t[-1]) if len(t) else np.array([])
            dt[dt < 0] = 0.0                                       # the jump in t_lap where a window crosses s = 0
            total_t, total_d = dt.sum(), len(w) * DISTANCE_GRID_STEP_M
            row["corner_time"], row["corner_dist"] = float(total_t), float(total_d)
            ph = w["phase"].to_numpy()
            for p in PHASES:
                m = ph == p
                key = p.replace(" ", "_")
                row[f"{key}_time"] = float(dt[m].sum())
                row[f"{key}_dist"] = float(m.sum() * DISTANCE_GRID_STEP_M)
                row[f"{key}_pct_time"] = 100 * row[f"{key}_time"] / total_t if total_t else np.nan
                row[f"{key}_pct_dist"] = 100 * row[f"{key}_dist"] / total_d if total_d else np.nan
            row["brake_total_time"] = row["braking_time"] + row["trail_braking_time"]
            row["brake_total_pct_time"] = 100 * row["brake_total_time"] / total_t if total_t else np.nan

            # coasting before the brake point, added to coasting 
            if e["entry_source"] in ("braking", "trail braking"):
                before = _stretch(lg, prev_a, e["s_entry"], length)
                thr_on = (before[THROTTLE_COL] > THROTTLE_OFF).to_numpy()
                if thr_on.any():
                    gap = before.iloc[int(np.where(thr_on)[0][-1]):]   # throttle release → brake point
                    gt = gap["t_lap"].to_numpy()
                    gdt = np.diff(gt, append=gt[-1])
                    gdt[gdt < 0] = 0.0
                    gm = (gap["phase"] == "coasting").to_numpy()
                    gap_t, gap_d = float(gdt.sum()), len(gap) * DISTANCE_GRID_STEP_M
                    row["coasting_time"] += float(gdt[gm].sum())
                    row["coasting_dist"] += float(gm.sum() * DISTANCE_GRID_STEP_M)
                    row["coasting_pct_time"] = 100 * row["coasting_time"] / (total_t + gap_t) if total_t + gap_t else np.nan
                    row["coasting_pct_dist"] = 100 * row["coasting_dist"] / (total_d + gap_d) if total_d + gap_d else np.nan

            # grip, exit throttle, steering corrections, exit slip
            row["peak_lat_g"] = float(w[LAT_COL].abs().max())
            row["max_exit_throttle"] = float(after_apex[THROTTLE_COL].max()) if len(after_apex) else np.nan

            sr = w["str_rate"].to_numpy()
            sign = np.sign(np.where(np.abs(sr) > DEADBAND_STEER_DEG_PER_M, sr, 0))
            sign = sign[sign != 0]
            row["steer_corrections"] = int((np.diff(sign) != 0).sum()) if len(sign) else 0

            front = after_apex[[FL_COL, FR_COL]].mean(axis=1)
            rear = after_apex[[RL_COL, RR_COL]].mean(axis=1)
            ok = front > MOTION_THRESHOLD_RAD_S
            row["exit_slip_pct"] = float(100 * (rear[ok].mean() / front[ok].mean() - 1)) if ok.any() else np.nan

            rows.append(row)
    return pd.DataFrame(rows)


METRIC_COLS = (["entry_from_apex", "peak_brake_bar", "min_speed", "min_speed_from_apex",
                "pickup_from_apex", "exit_speed", "corner_time", "corner_dist",
                "brake_total_time", "brake_total_pct_time", "peak_lat_g", "max_exit_throttle",
                "steer_corrections", "exit_slip_pct"]
               + [f"{p.replace(' ', '_')}_{k}" for p in PHASES
                  for k in ("time", "dist", "pct_time", "pct_dist")])


def metrics_vs_ref(cm: pd.DataFrame, ref: int):
    """
    Adds <metric>_ref and <metric>_diff (lap − reference) for every metric
    
    """
    r = cm[cm["lap"] == ref].set_index("corner")[METRIC_COLS]
    out = cm.join(r, on="corner", rsuffix="_ref")
    for col in METRIC_COLS:
        out[f"{col}_diff"] = out[col] - out[f"{col}_ref"]
    return out


## corner study
def group_times(grid: pd.DataFrame, apexes: pd.DataFrame, entries: pd.DataFrame, zones: pd.DataFrame,
                gaps: pd.DataFrame, laps_used: list[int], groups: dict[str, list[int]],
                length: float):
    """
    one row per (group, lap): time in the group's window.
    window, same for every lap: earliest entry of the first corner across laps -> zone end of the last corner
    gap = True if a NaN run of s starts inside the window in that lap.
    """
    apex_s = apexes[apexes["is_main"]].set_index("corner")["s"]
    exits = corner_exits(apexes, entries, zones, laps_used, length)
    e = entries[entries["lap"].isin(laps_used)]
    g_ok = gaps.dropna(subset=["s_start"])
    gap_s = {lap: gl["s_start"].tolist() for lap, gl in g_ok.groupby("lap")}
    stint = grid.groupby("lap")["stint"].first()
    lap_grids = {lap: lg.sort_values("s") for lap, lg in grid[grid["lap"].isin(laps_used)].groupby("lap")}

    rows = []
    for name, corners in groups.items():
        first, last = corners[0], corners[-1]
        e1 = e[e["corner"] == first]
        back = np.mod(apex_s[first] - e1["s_entry"], length).to_numpy()   # metres before the apex
        s0 = float(e1["s_entry"].iloc[int(np.argmax(back))])              # earliest entry
        s1 = float(exits[last])
        span = np.mod(s1 - s0, length)
        for lap, lg in lap_grids.items():
            gap = any(np.mod(gs - s0, length) <= span for gs in gap_s.get(lap, []))
            rows.append({"group": name, "lap": int(lap), "stint": int(stint[lap]),
                         "s_start": s0, "s_end": s1,
                         "time": _window_time(lg, s0, s1), "gap": gap})
    return pd.DataFrame(rows)

def pick_laps(gt: pd.DataFrame, ref: int):
    """
    Per group: the reference lap, the fastest lap, and the slowest lap of each stint.
    Laps with a gap in the window and warmup laps are left out of the picks
    """
    rows = []
    for name, g in gt.groupby("group", sort=False):
        ok = g[~g["gap"] & ~g["lap"].isin(WARMUP_LAPS)]
        ref_time = float(g.loc[g["lap"] == ref, "time"].iloc[0])
        picks = [("ref", ref), ("fastest", int(ok.loc[ok["time"].idxmin(), "lap"]))]
        for st, gs in ok.groupby("stint"):
            picks.append((f"slowest stint {st}", int(gs.loc[gs["time"].idxmax(), "lap"])))
        for role, lap in picks:
            t = float(g.loc[g["lap"] == lap, "time"].iloc[0])
            rows.append({"group": name, "role": role, "lap": lap, "time": t, "delta": t - ref_time})
    return pd.DataFrame(rows)

def apply_exclusions(gt: pd.DataFrame, exclusions: dict[str, list[int]]) -> pd.DataFrame:
    """
    marks (group, lap) pairs as gap = True so pick_laps leaves them out.
    exclusions = {group: [laps]} from config (MANUAL_EXCLUSIONS): laps whose t_lap is broken
    inside that group's window (logger-gap pattern), found with the time vs speed check.
    returns a copy, gt is not changed
    """
    out = gt.copy()
    for group, bad in exclusions.items():
        out.loc[(out["group"] == group) & out["lap"].isin(bad), "gap"] = True
    return out


# metrics shown per group in the corner study, positions are metres from the apex (- = before)
SHOW = ["entry_from_apex", "peak_brake_bar", "brake_total_time", "coasting_time", "coasting_dist",
        "min_speed", "min_speed_from_apex", "pickup_from_apex", "max_exit_throttle", "exit_speed",
        "steer_corrections", "exit_slip_pct", "corner_time"]


def group_metrics(name: str, cm: pd.DataFrame, picks: pd.DataFrame, gt: pd.DataFrame,
                  grid: pd.DataFrame, groups: dict[str, list[int]], main_apex: pd.Series,
                  length: float) -> tuple[pd.DataFrame, dict[int, float]]:
    """
    metrics of the picked laps of one group: metrics as rows, (corner, lap) as columns,
    plus speed at the window start, group time and group delta vs the reference.
    also returns where each main apex of the group is, in metres from the window start.
    """
    p = picks[picks["group"] == name].drop_duplicates("lap")
    lap_list = p["lap"].tolist()
    m = cm[cm["lap"].isin(lap_list) & cm["corner"].isin(groups[name])]
    out = m.set_index(["corner", "lap"])[SHOW].T

    s0 = float(gt.loc[gt["group"] == name, "s_start"].iloc[0])
    for lap in lap_list:
        lg = grid[grid["lap"] == lap].sort_values("s")
        u0 = float(np.interp(s0, lg["s"], lg[U_COL]))
        r = p[p["lap"] == lap].iloc[0]
        for c in groups[name]:
            out.loc["speed_at_window_start", (c, lap)] = u0
            out.loc["group_time", (c, lap)] = r["time"]
            out.loc["group_delta_vs_ref", (c, lap)] = r["delta"]

    apex_pos = {int(c): float(np.mod(main_apex[c] - s0, length)) for c in groups[name]}
    return out.round(2), apex_pos
