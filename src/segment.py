
from src.config import ( PIT_MIN_S, PROC_DATA, MOTION_THRESHOLD_RAD_S, STANDSTILL_MIN_S, DISTANCE_GRID_STEP_M, CURV_SMOOTH_WINDOW_M, 
                        K_STRAIGHT, MIN_STRAIGHT_M, K_CORNER, K_LINK, MAX_LINK_M, PHASE_SMOOTH_M, THROTTLE_OFF, THROTTLE_FULL, BRAKE_OFF_BAR, 
                        STEER_ZERO_DEG, DEADBAND_BRAKE_BAR_PER_M, DEADBAND_THROTTLE_PER_M, DEADBAND_STEER_DEG_PER_M, NEEDED_CHANNELS, SEGMENT_FILES )
import pandas as pd
from src.preprocess import load_processed
import numpy as np
from src.loader import load_centerline

S_COL = "vehicle_curvilinear_coordinates__s"
FL_COL = "front_angular_velocity__fl"
FR_COL = "front_angular_velocity__fr"
U_COL = "vehicle_speed__u"

THROTTLE_COL = "pedal_throttle__throttle"
BRAKE_COL = "pedal_brakes_pressure__front"
STEER_COL = "steer_angle__angle"




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



# distance grid

def nan_runs(df: pd.DataFrame, col: str):
    t = df["t_s"].to_numpy()
    is_nan = df[col].isna().to_numpy().astype(int)

    # +1 where a NaN run starts, -1 just after it ends
    edges = np.diff(np.concatenate([[0], is_nan, [0]]))
    first = np.flatnonzero(edges == 1)    
    last = np.flatnonzero(edges == -1) - 1

    before = np.clip(first - 1, 0, len(t) - 1)
    after = np.clip(last + 1, 0, len(t) - 1)

    return pd.DataFrame({
        "t_start": t[first], # first NaN
        "t_end": t[last], # last NaN
        "t_before": t[before], # last known value before the run
        "t_after": t[after], # first known value after the run
        "duration_s": t[last] - t[first],
    })

def build_distance_grid(df: pd.DataFrame, laps: pd.DataFrame, standstills: pd.DataFrame, track_length: float, step: float = DISTANCE_GRID_STEP_M,
                        channels: list = NEEDED_CHANNELS, save: bool = True):
    """Every lap (pit laps excluded) on a fixed distance grid, one row per lap and grid point."""
    valid_laps = laps[~laps["has_pit"]] # exclue pit lap
    stops = standstills[standstills["type"] == "stop"] # stops that are not a pit, such as blue flags
    grid = np.arange(0, track_length, step) # grid that is DISTANCE_GRID_STEP_M apart 

    lap_tables = []
    for lap in valid_laps.itertuples():
        # 1. exclude the rows that were part of a stop
        lap_rows = df[df["lap"] == lap.lap]
        t = lap_rows["t_s"].to_numpy()
        in_stop = np.zeros(len(lap_rows), dtype=bool)
        for start, end in zip(stops["start_s"], stops["end_s"]):
            in_stop |= (t >= start) & (t <= end)
        rows = lap_rows[~in_stop]

        # 2. interpolate the time (t_s) on the grid
        known_s = rows[rows[S_COL].notna()]
        t_grid = np.interp(grid, known_s[S_COL], known_s["t_s"])

        # 3. time since lap start
        t_lap = t_grid - lap.t_start

        table = {"lap": lap.lap, "stint": lap.stint, "s": grid,
                 "t_s": t_grid, "t_lap": t_lap}


        for channel in channels:
            known = rows[rows[S_COL].notna() & rows[channel].notna()] # rows ehere both s and the channel are not nan
            if len(known) < 2: # no known rows
                table[channel] = np.full(len(grid), np.nan) # fill it all with nan
                continue
            # channel's values at each grid distance, interpolated for unknown ones
            values = np.interp(grid, known[S_COL], known[channel]) 

            # set the grid points of the channel that fall in nan runs to nan
            runs = nan_runs(rows, channel)
            for before, after in zip(runs["t_before"], runs["t_after"]):
                values[(t_grid > before) & (t_grid < after)] = np.nan

            # add the cleaned column to the table
            table[channel] = values

        lap_df = pd.DataFrame(table)

        # now find the nan runs of s
        s_runs = nan_runs(rows, S_COL)
        # set the grid points that fall inside the nan runs to nan
        for before, after in zip(s_runs["t_before"], s_runs["t_after"]):
            lap_df.loc[(t_grid > before) & (t_grid < after), channels] = np.nan


        lap_tables.append(lap_df)

    # 5. stack all laps and save
    grid_df = pd.concat(lap_tables, ignore_index=True)
    if save:
        grid_df.to_parquet(PROC_DATA / "laps_distance.parquet", index=False)
    return grid_df
    
    
def smooth_circular(values: np.ndarray, window: int):
    """
    Rolling mean over 'window' points, 
    wrapping around the end of the loop
    
    """
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    return np.convolve(padded, np.ones(window) / window, mode="valid")

def runs_of(mask: np.ndarray):
    """
    Number each run of consecutive equal values first and 
    last run joined (track is a loop)
    
    """
    run_id = np.concatenate([[0], np.cumsum(mask[1:] != mask[:-1])])
    if mask[0] == mask[-1] and run_id[-1] != 0:
        run_id[run_id == run_id[-1]] = 0
    return run_id

# trun the work in notebook 7 to a function
def find_corners(cl: pd.DataFrame, window: int = CURV_SMOOTH_WINDOW_M,
                 k_straight: float = K_STRAIGHT, min_straight_m: float = MIN_STRAIGHT_M,
                 k_corner: float = K_CORNER, k_link: float = K_LINK, max_link_m: float = MAX_LINK_M):

    
    """
    
    smoothed curvature
    straights and apexes (grouped into corners) on the centerline
    
    """

    cl = cl.copy() # centerline
    cl["curvature_smooth"] = smooth_circular(cl["curvature"].to_numpy(), window) # smooth the curvature
    curv = cl["curvature_smooth"].abs().to_numpy()
    n = len(curv)

    # straights: low curvature for at least min_straight_m 
    low = curv < k_straight
    low_runs = runs_of(low)
    cl["is_straight"] = low & (np.bincount(low_runs)[low_runs] >= min_straight_m)

    # apexes: one per run of high curvature, at its sharpest point
    high = curv > k_corner
    cl["high_run"] = runs_of(high)
    apex_idx = cl[high].groupby("high_run")["curvature_smooth"].apply(lambda c: c.abs().idxmax())
    apexes = cl.loc[apex_idx].sort_values("s").reset_index()      # "index" = centerline position
    apexes = apexes.rename(columns={"index": "pos"})
    apexes["direction"] = np.where(apexes["curvature_smooth"] > 0, "L", "R")

    # group apexes into corners: same direction, close, and the track keeps turning between them
    def same_corner(a, b):
        if a["direction"] != b["direction"]:
            return False
        i, j = int(a["pos"]), int(b["pos"]) # position of each apex in m on the track
        # if j and i are in the same lap (pos f j is >= pos of i) then
        # if j belongs to the next lap then take cuv i:805 and  then curv 0:j
        between = curv[i:j + 1] if j >= i else np.concatenate([curv[i:], curv[:j + 1]]) 
        # return true if teh distance <= 35 m (the %n handles the crossing at start/finish)
        # and if the lowest curvature between them stays at least k_link so the car doesnt staighten
        # else, they are separate corners
        return (j - i) % n <= max_link_m and between.min() >= k_link

    corner_id = [0]
    for k in range(1, len(apexes)):
        same = same_corner(apexes.iloc[k - 1], apexes.iloc[k]) # this apex and the one before it
        corner_id.append(corner_id[-1] if same else corner_id[-1] + 1) # assign the corner number
    corner_id = np.array(corner_id)
    if len(apexes) > 1 and same_corner(apexes.iloc[-1], apexes.iloc[0]):   # last and first apex in the same corner
        corner_id[corner_id == corner_id[-1]] = corner_id[0] # assign it the corner of the new lap
    apexes["corner"] = pd.factorize(corner_id)[0] + 1 # reumbering corners

    # highest curv apex in a corner and make it the main apex of the corner
    main_idx = apexes.groupby("corner")["curvature_smooth"].apply(lambda c: c.abs().idxmax())
    apexes["is_main"] = apexes.index.isin(main_idx)

    return cl.drop(columns="high_run"), apexes[["corner", "pos", "s", "x", "y", "curvature_smooth", "direction", "is_main"]]


# now it's time we define the zones in the track

def run_positions(run_id: np.ndarray, r: int):
    """
    
    positions of run r in driving order 
    (a run that wraps the line starts before 805 and continues from 0)
    
    """
    pos = np.flatnonzero(run_id == r)
    breaks = np.flatnonzero(np.diff(pos) > 1)
    if len(breaks): # wraps: put the tail end first
        pos = np.concatenate([pos[breaks[0] + 1:], pos[:breaks[0] + 1]])
    return pos



def build_zones(cl: pd.DataFrame, apexes: pd.DataFrame):
    """
    
    split the track into straight, corner and kink zones
    
    returns (centerline with a zone column, zones table)
    
    """
    curv = cl["curvature_smooth"].abs().to_numpy()
    s = cl["s"].to_numpy()
    is_straight = cl["is_straight"].to_numpy()
    zone_of = np.full(len(cl), -1)
    zones = []

    def add_zone(positions, ztype, corner=None):
        z = len(zones)
        zone_of[positions] = z
        row = {"type": ztype,
               "corner": corner,
               "s_start": s[positions[0]],
               "s_end": s[positions[-1]],
               "contains_zero": 0 in positions  # contains the start/finish line
            }
        if corner is not None:
            main = apexes[(apexes["corner"] == corner) & apexes["is_main"]].iloc[0] # find main apex
            row.update(direction=main["direction"], s_apex=main["s"]) # adds corner direction and its apex's position
        zones.append(row)

    run_id = runs_of(is_straight)
    for r in np.unique(run_id):
        pos = run_positions(run_id, r)
        # straight runs
        if is_straight[pos[0]]: # every meter in a run has the same value so checking teh first one is enough
            add_zone(pos, "straight")
            continue # skip to next 

        # for non straight runs
        # section: which apexes fall inside it, in driving order
        rank = {p: k for k, p in enumerate(pos)}
        inside = apexes[apexes["pos"].isin(rank)].copy()
        inside["rank"] = inside["pos"].map(rank)
        inside = inside.sort_values("rank")
        corners = list(dict.fromkeys(inside["corner"])) # unique, in driving order

        if not corners: # no apex: kink 
            add_zone(pos, "kink")
            continue

        # split between consecutive corners at the lowest curvature between them
        start = 0
        for c_now, c_next in zip(corners[:-1], corners[1:]):
            last_now = inside.loc[inside["corner"] == c_now, "rank"].max()
            first_next = inside.loc[inside["corner"] == c_next, "rank"].min()
            split = last_now + np.argmin(curv[pos[last_now:first_next + 1]])
            add_zone(pos[start:split + 1], "corner", c_now)
            start = split + 1
        add_zone(pos[start:], "corner", corners[-1])

    # number zones in driving order, starting with the zone that contains s = 0
    zones = pd.DataFrame(zones)
    order = zones.assign(key=np.where(zones["contains_zero"], -1, zones["s_start"])).sort_values("key").index
    renumber = {old: new + 1 for new, old in enumerate(order)}
    zones = zones.loc[order].reset_index(drop=True)
    zones.insert(0, "zone", np.arange(1, len(zones) + 1))

    cl = cl.copy()
    cl["zone"] = [renumber[z] for z in zone_of]
    return cl, zones.drop(columns="contains_zero")


# labeling the zones
def label_zones(grid_df: pd.DataFrame, cl: pd.DataFrame, zones: pd.DataFrame):
    """
    give every grid row the zone, zone type and corner of the centerline metre it falls in
    
    """
    grid_df = grid_df.copy()
    pos = np.floor(grid_df["s"].to_numpy()).astype(int) % len(cl)
    grid_df["zone"] = cl["zone"].to_numpy()[pos]
    return grid_df.merge(zones[["zone", "type", "corner"]], on="zone", how="left")


# phases

def add_rates(grid_df: pd.DataFrame, smooth_m: float = PHASE_SMOOTH_M, step: float = DISTANCE_GRID_STEP_M):
    """
    abs_steer plus the per-metre rates of throttle, brake and |steering|,
    computed per lap after a centred rolling mean over smooth_m metres
    """
    w = max(1, int(round(smooth_m / step)))  # smoothing window in grid points
    out = []
    for lap, d in grid_df.groupby("lap"):
        d = d.sort_values("s").copy()
        d["abs_steer"] = d[STEER_COL].abs()
        s = d["s"].to_numpy()
        for col, key in [(THROTTLE_COL, "thr"), (BRAKE_COL, "brk"), ("abs_steer", "str")]:
            smooth = d[col].rolling(w, center=True, min_periods=1).mean()
            d[f"{key}_rate"] = np.gradient(smooth.to_numpy(), s)  # change per metre
        out.append(d)
    return pd.concat(out, ignore_index=True)


def label_phases(grid_df: pd.DataFrame,
                 throttle_off: float = THROTTLE_OFF, throttle_full: float = THROTTLE_FULL,
                 brake_off: float = BRAKE_OFF_BAR, steer_zero: float = STEER_ZERO_DEG,
                 db_throttle: float = DEADBAND_THROTTLE_PER_M,
                 db_brake: float = DEADBAND_BRAKE_BAR_PER_M,
                 db_steer: float = DEADBAND_STEER_DEG_PER_M):
    """
    one phase per grid point, first matching rule wins:
    ?? -> trail braking -> braking -> full throttle -> coasting -> lift -> throttle on
    points with any NaN input get phase NaN. needs the columns from add_rates
    """
    thr, brk, st = grid_df[THROTTLE_COL], grid_df[BRAKE_COL], grid_df["abs_steer"]

    # levels: where the pedal / wheel is
    brake_on = brk > brake_off
    thr_on = thr > throttle_off
    full = thr >= throttle_full
    straight = st <= steer_zero

    # rates: whether it is moving (changes inside the deadband count as steady)
    thr_rising = grid_df["thr_rate"] > db_throttle
    thr_falling = grid_df["thr_rate"] < -db_throttle
    brk_rising = grid_df["brk_rate"] > db_brake
    brk_falling = grid_df["brk_rate"] < -db_brake
    st_rising = grid_df["str_rate"] > db_steer

    any_nan = grid_df[[THROTTLE_COL, BRAKE_COL, "abs_steer",
                       "thr_rate", "brk_rate", "str_rate"]].isna().any(axis=1)

    conditions = [
        any_nan,
        brake_on & brk_rising & thr_on & thr_rising,           # ?? both pedals rising
        brake_on & (~straight | (brk_falling & st_rising)),    # trail braking
        brake_on,                                              # braking
        full,                                                  # full throttle
        ~thr_on & ~brake_on,                                   # coasting
        thr_falling & ~brake_on,                               # lift
    ]
    choices = ["__nan__", "??", "trail braking", "braking", "full throttle",
               "coasting", "lift"]

    grid_df = grid_df.copy()
    phase = np.select(conditions, choices, default="throttle on")
    grid_df["phase"] = pd.Series(phase, index=grid_df.index).replace("__nan__", np.nan)
    return grid_df


def run_segmentation(save: bool = True):
    """
    run the whole segmentation: standstills, stints, laps, distance grid,
    corners and zones, then rates and phases on the grid, and save the outputs.
    """
    df = load_processed()
    centerline = load_centerline()
    track_length = centerline.attrs["length"]

    # stints and laps (time domain)
    standstills = classify_standstills(df)
    df, stints = define_stints(df, track_length)
    df, laps, gaps = segment_laps(df, standstills, track_length)

    # distance grid
    grid_df = build_distance_grid(df, laps, standstills, track_length, save=save)

    # corners and zones
    cl, apexes = find_corners(centerline)
    cl, zones = build_zones(cl, apexes)
    grid_df = label_zones(grid_df, cl, zones)

    # phases
    grid_df = label_phases(add_rates(grid_df))

    outputs = {"grid": grid_df, "laps": laps, "gaps": gaps, "zones": zones,
               "apexes": apexes, "stints": stints, "standstills": standstills}

    if save:
        PROC_DATA.mkdir(parents=True, exist_ok=True)
        for name, table in outputs.items():
            table.to_parquet(PROC_DATA / SEGMENT_FILES[name], index=False)

    return outputs


def load_segmented():
    """
    
    read the segmentation outputs, running the segmentation if any file is missing
    
    """
    paths = {name: PROC_DATA / file for name, file in SEGMENT_FILES.items()}
    if not all(p.exists() for p in paths.values()):
        return run_segmentation()
    return {name: pd.read_parquet(p) for name, p in paths.items()}





    
    
