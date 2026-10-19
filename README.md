# recruiting-sw-telemetry

Offline analysis of a real Formula Student track session, built as an interactive Streamlit app.

The data is one Endurance run logged by **Hydra**, E-Agle TRT's 2024 electric car, at the **Varano** circuit on 8 September 2024. The log comes straight off the car: every CAN message on its own clock and at its own rate, no lap list, no annotations. The project turns it into something a race engineer can use after a session, and answers three questions:

1. **Where** is the car driven differently from the best lap?
2. **How much** lap time is that worth?
3. **What** would the driver have to change to get it back?

The session has two stints driven by two different drivers on the same car and setup, so the differences between them are differences in driving style.

## Contents

- [Running the app](#running-the-app)
- [Repository layout](#repository-layout)
- [Method](#method)
  - [1. Exploring the log](#1-exploring-the-log)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Segmentation](#3-segmentation)
  - [4. Analysis](#4-analysis)
- [Corner metrics: definitions](#corner-metrics-definitions)
- [The app](#the-app)
- [Assumptions](#assumptions)
- [Limitations](#limitations)

## Running the app

Tested with Python 3.12.

**1. Get the data.** The log is not in this repo. Download it from [eagletrt/recruiting-sw](https://github.com/eagletrt/recruiting-sw/tree/master/telemetry/project_3) and copy the contents of `2024_09_08_14_38_41_ENDURANCE_run1/` into `data/raw/`. `centerline.json` must sit directly in `data/raw/`. The `.csv.gz` files can be anywhere below it, since they are found by name with a recursive search.

**2. Install.** From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` contains `-e .`, so `src` is installed as a package and can be imported from the notebooks, the scripts and the app.

**3. Run.** From the repo root (so the theme in `.streamlit/config.toml` is picked up):

```bash
streamlit run app/main.py
```

The whole pipeline runs from this one command. On the first run the app builds whatever is missing, in order:

| Step | What it does | Writes | Time |
|---|---|---|---|
| `compute_t0` | finds the session's time reference | `session_cache.json` | ~3.5 s |
| `run_preprocessing` | section 2 | `data/processed/session.parquet` | ~11 s |
| `run_segmentation` | section 3 | 7 parquet tables in `data/processed/` | ~4 s |
| section 4 computations | corner entries, times, metrics, lap picks | in memory | ~6 s |

About 25 s from raw data only, about 6 s once the processed files exist. After that, results are cached for as long as the Streamlit server runs.

Outputs are only rebuilt when they are missing. If the raw data changes, delete `data/processed/*.parquet` and run:

```bash
python scripts/clear_cache.py
```

The data folder must hold one session only. Files added later must come from the same session.

## Repository layout

```
├── config.yaml          all settings: paths, thresholds, corner groups, app keys
├── app/
│   ├── main.py          entry point and navigation
│   ├── data.py          loads the data and runs the analysis for the app (cached)
│   ├── plots.py         all Plotly figures
│   ├── views/           one file per page
│   └── content/         write-ups, recommendations and limitations shown in the app
├── src/
│   ├── config.py        reads config.yaml
│   ├── session.py       file discovery and the session time reference (t0)
│   ├── loader.py        load one raw file by short name, load the centerline
│   ├── profiler.py      data profiling: rates, gaps, NaNs, logger-wide gaps
│   ├── utils.py         exploration helpers: standstills, laps
│   ├── preprocess.py    section 2: offsets, common time base, filtering
│   ├── segment.py       section 3: stints, laps, distance grid, corners, phases
│   └── analysis.py      section 4: computations only, no plotting
├── notebooks/           01 to 09, one per step (see below)
├── output/logs/issues.csv   data-quality log
├── scripts/clear_cache.py
└── data/raw, data/processed   (gitignored)
```

The data layer (`session`, `loader`, `preprocess`, `segment`), the analysis layer (`analysis`) and the UI layer (`app/`) are separate. Plotting never lives in `src/`. Imports go one way: `config ← session ← loader ← profiler`. No setting is hardcoded: session-specific choices and every threshold live in `config.yaml`, and values that come from the data (such as the track length) are passed in.

The notebooks are where each decision was worked out and checked on plots:

| Notebook | Topic |
|---|---|
| `01_profile` | profiling the raw files for inconsistencies |
| `02_session` | session structure: standstills, stints, pit stop, logger-wide gaps |
| `03_standstill_analysis` | sensor offsets and noise while the car is at rest |
| `04_sign_conventions` | left/right agreement across signals |
| `05_laps` | lap boundaries, lap times, lap status |
| `06_preprocess` | the cleaned data and the filter cutoff choices |
| `07_segment` | distance grid, corners, corner phases |
| `08_analyze` | time delta, delta map, corner entries, corner times, metrics |
| `09_corner_study` | corner-by-corner comparison, write-ups, recommendations |

## Method

### 1. Exploring the log

The log is 2207 s long. Each CAN message has its own `_timestamp` (Unix, microseconds) and its own rate, from ~7.6 Hz (GPS) to ~235 Hz (IMU).

**Time reference.** `t0` is the earliest timestamp across all files, and every file is expressed in seconds since `t0` (`t_s`). It is computed once and cached.

**Gaps.** A gap in a file is a step between two consecutive samples of at least `gap_factor × T`, where `T` is that file's median step (`gap_factor` = 3). Gaps that appear in every file at the same time are logger-wide gaps. Since the slowest file (GPS) can't show short gaps, only gaps longer than `gap_factor ×` the largest `T` are considered, they are matched by overlapping intervals against GPS, and merged as the union of the matched intervals.

**Standstills.** The car is stopped when both front wheel speeds are below 1 rad/s (about 0.19 m/s) for at least 3 s. 0 can't be used because the sensors read noise around 0. 1 rad/s is low enough that a car being pushed still counts as moving. 3 s is there because a locked wheel can read 0 before the car stops: peak braking is about 1.2 g, so stopping from 23 m/s takes about 2 s, plus margin.

**Session structure.** The standstills split the session into a wait on the grid, two stints and a pit stop between them. A standstill counts as a pit stop if it lasts at least 60 s, well above the ~15 s stops on track and the 34 s wait at the start, and still short enough for a fast driver change. Early in stint 1 the car stops three times for ~15 s each (see [Assumptions](#assumptions)).

**Sensor offsets.** Measured on the standstill windows:
- IMU longitudinal acceleration reads ~-0.07 g at rest, the same at every location, so it's a mounting angle or bias, and it is subtracted.
- Gyro offsets are subtracted, using the average of the standstill-window means.
- IMU lateral (~-0.02 g) and vertical (~-0.97 g instead of -1 g) offsets are small and neglected.
- Brake, power, motor current, wheel speeds and throttle offsets are negligible. Steering, estimator outputs and GPS were not included.

**Sign conventions.** Checked on a left corner: centerline curvature, lateral offset `n`, steering, yaw rate, lateral acceleration and estimator heading are all positive to the left. No flips needed.

**Laps.** A lap boundary is where the distance along the track `s` drops by more than half the track length (805 m) while the front wheels are turning, and at least a minimum lap time (track length / session max speed) after the previous boundary. This assumes the car only crosses the line driving forward. The session has 23 laps: laps 1 to 13 and the pit lap 14 in stint 1, laps 15 to 23 in stint 2.

Every issue found in the data is logged by hand in `output/logs/issues.csv` (file, column, issue, evidence, time window, fix, status) and shown in the app. The profiler only produces the tables that feed it, it never writes to the log.

### 2. Preprocessing

**Channels used.** 14 files: driver inputs (throttle, brake front/rear, steering), IMU acceleration and angular rate, front wheel speeds, rear motor speeds and currents, HV power, and the on-board state estimator's speed, position and curvilinear coordinates. GPS is not loaded because the estimator outputs already fuse it.

**Common time base.** All channels are resampled onto one 250 Hz grid, at least as fast as the fastest channel (IMU, ~235 Hz). The grid starts at the latest first sample among the files and ends at the earliest last sample, which cuts nothing from any lap.

**Missing data.** Gaps of at least `gap_factor × T` stay NaN and are never interpolated across. Shorter drops are interpolated.

**Filtering.** Zero-phase Butterworth low-pass (order 4, `sosfiltfilt`), run separately on each stretch between NaN runs, after resampling. Cutoffs were chosen by looking at each signal's spectrum (Welch, 4 s windows, on its longest NaN-free driving stretch, one per stint) together with how fast the event each signal is used for happens: an event lasting T seconds needs content up to about 1/T Hz. The cutoff sits above that and below the noise and vibration bumps.

| Signal | Cutoff | Why |
|---|---|---|
| Brake pressure front/rear | 15 Hz | brake applied in ~0.1 s needs ~10 Hz, bump at 25–35 Hz |
| Front wheel speeds | 5 Hz | limited by the car's inertia, bump at ~10 Hz |
| IMU acceleration and angular rate | 10 Hz | the body responds in ~0.1–0.2 s, above that is vibration |
| Throttle, steering | not filtered | smooth spectra, no noise region |
| Estimator outputs, inverter `_filt` channels | not filtered | already smoothed at the source |
| HV power | not filtered | native 20 Hz, not used for fast events |

A rule based on 95% of cumulative power was tried and rejected: the power is dominated by large slow swings, so it gave cutoffs of ~0.5–1.2 Hz and would have removed the small fast features that matter, like brake onset. The final cutoffs were checked with filtered vs unfiltered overlays on a braking zone and a corner exit: no time shift, peaks and onsets kept, noise removed.

The result is `data/processed/session.parquet`: one clean, aligned table at 250 Hz.

### 3. Segmentation

**Stints** are split by the pit stop, not by every standstill, so the three short stops in stint 1 don't split it. The pit belongs to the stint before it.

**Laps** are found again on the 250 Hz data with the same three rules as in section 1, using only rows where `s` and both front wheel speeds are known. A lap runs between two consecutive boundaries, so the stretch before the first crossing and after the last one are not laps. Each lap gets flags (labels only, nothing is removed):
- `has_stop`: contains a standstill (lap 2, the three short stops)
- `has_pit`: contains the pit stop (lap 14)
- `has_nan_s`: `s` has at least one NaN inside the lap. This includes the estimator's short NaN runs of 50–200 ms, so most laps have it.

Only the pit lap is invalid. A NaN run doesn't throw away a whole lap (that would cost too many of 23 laps for an analysis that works corner by corner), its position is recorded in metres instead.

**Distance grid.** Comparing laps in time is useless because they drift apart after the first corner, so every lap is interpolated onto a fixed `s` grid with a 0.3 m step: 2684 points per lap, the same for every lap. 0.3 m keeps the detail of braking points and apexes. Time is kept as two channels, `t_s` and `t_lap` (time since the lap started), so time can still be compared at any point.
- Per channel, rows where `s` or that channel is NaN are dropped, the rest is interpolated onto the grid, and grid points whose time falls inside that channel's own NaN runs are set back to NaN. Points where `s` itself was NaN are blanked in every channel, since the position is unknown there.
- Seams: a lap starts at the first sample after `s` wraps (e.g. 804.8 → 0.2), with no interpolation of the exact crossing. The ~4 ms between one lap's last sample and the next lap's first is accepted. Grid ends take the nearest value.
- Lap 2's stopped rows are removed before interpolating. The pit lap is left off the grid.

**Corners** are fixed zones on the track, the same for every lap, defined from the centerline curvature only, not from the driver's behaviour:
1. Curvature is smoothed with a 5 m rolling mean that wraps around the loop (single-metre spikes are not corners).
2. A straight is |curvature| below 0.021 1/m for at least 20 m. Everything else is a corner section.
3. Apexes are peaks of smoothed |curvature| above 0.0385 1/m. The apex is the point of maximum curvature.
4. Two consecutive apexes belong to the same corner (a double apex) if they turn the same way, the curvature between them never drops below 0.025 1/m, and they are at most 35 m apart. Each corner's main apex is its sharpest one.
5. Consecutive corners inside one section are split at the lowest curvature between them. A section with no apex would be a "kink" (there are none on this layout).

Zones are numbered in driving order from the start/finish line, and every grid point is labelled with its zone, type and corner.

**Corner phases.** Each grid point is labelled with a phase from the pedals, the steering and how fast they change per metre:

| Phase | In plain words |
|---|---|
| braking | on the brakes |
| trail braking | still on the brakes while turning in |
| full throttle | throttle at full travel |
| coasting | neither pedal |
| lift | slowing by easing off the throttle instead of braking |
| throttle on | throttle applied |

The first matching rule wins (exact rules in `src/segment.py`). The thresholds were read from the data:
- **Off levels**, from histograms of the raw signals at the edge of the "untouched" spike: throttle off < 0.01, brake off < 0.5 bar. Full throttle = 0.95 (full pedal travel), even though the drivers rarely reach it.
- **Steering ≈ 0** = |steering| ≤ 10 deg, which covers ~90% of steering readings when the car really goes straight (|lateral acc| < 0.03 g) in both stints. The steering zero differs by ≤ ~1 deg between stints, so one set of thresholds is used for both.
- **Rising, falling or steady** is measured per metre on signals smoothed over 2.4 m (raw differences turn sensor jitter into large rates). The deadband, the band around zero where a rate counts as steady, is 2.5 deg/m for steering (95th percentile of the steering rate on straights), 0.02 /m for throttle and 1 bar/m for brake (from the spread of their rate distributions, calibrated on steering).

### 4. Analysis

**Which laps are compared.** Lap 2 is excluded from time comparisons because its stopping time stays in its lap time. Lap 1 is kept but labelled warm-up (cold tyres). The pit lap is not on the grid. The **reference lap** is the fastest comparable lap of the session: lap 10.

**Time delta.** The time difference between two laps at each point of the track is the difference of their `t_lap` at the same `s`. It is drawn on the centerline, coloured by how fast time is gained or lost (ms per metre), because the driven position is distorted around NaN runs of `s`.

**Corner entry, per lap.** Between the previous corner's main apex and this one's, the entry is the start of the longest braking run, else the longest trail braking run, else the lift run closest to the apex, else the start of the corner zone. Each entry records which of these it came from. Runs shorter than 1 m are ignored and interruptions shorter than 1 m are bridged.

**Corner exit, same for every lap.** The end of the corner zone, or the earliest entry into the next corner across all laps if that comes first, so the next corner's braking is never counted in this one.

**Corner time loss.** Measured from the earlier of the two laps' entries to the end of the corner zone, so both laps cover the same stretch. A corner is invalid for a lap only if a window edge falls inside a NaN run of `s` (a NaN run inside the window cancels out).

**Corner study.** Six groups of corners are studied in detail: C1 (wide, fast corner), the esses (C2–C5), C20 (90 degrees corner), the two hairpins C24 and C27, and the snail (C25–C26). For each group the window is the same for every lap: from the earliest entry into its first corner across all laps to the exit of its last corner. Four laps are picked per group: the reference, the fastest lap on that group, and the slowest lap of each stint. Warm-up laps and laps with a NaN run of `s` starting inside the window are left out.

**Catching broken laps.** During some logger gaps `s` barely moves, then catches up, so `t_lap` at a given `s` is locally wrong even though the lap time is right. The NaN rule doesn't catch this, so every group is checked by comparing its time with a time rebuilt from speed (`Σ 0.3 / u` over the window). The two never match exactly (the centerline distance isn't the driven distance), so this is only used to spot laps that are clearly off, checked by eye. Laps excluded this way: lap 21 on C20, laps 3, 7, 9 and 23 on C27.

**Time at stake.** For each group and each driver, the time of that stint's slowest lap minus the reference. This is "up to" how much there is to gain, not a promise.

The per-corner, per-driver write-ups and recommendations are in the app, under **Opportunities** and **Corner deep-dive**.

## Corner metrics: definitions

Notation: `a` = the corner's main apex, `s_entry` = the lap's own entry, `s_exit` = the corner's exit point, window = the lap's grid points from `s_entry` to `s_exit` in driving order. Positions "from apex" are signed metres from `a` (negative = before). `dt` = difference of `t_lap` between consecutive grid points, step `Δs` = 0.3 m.

| Metric | Definition |
|---|---|
| `entry_from_apex` | `s_entry − a`, with `entry_source` = braking, trail braking, lift or zone |
| `peak_brake_bar` | max front brake pressure from `s_entry` to `a` |
| `min_speed` | min speed in the window, `min_speed_from_apex` = its position − `a` |
| `pickup_from_apex` | start of the last throttle-on run that lasts until `s_exit` (and doesn't start at the first window point), − `a`. NaN if the throttle never went off or is off at the exit |
| `exit_speed` | speed at `s_exit` |
| `corner_time` | Σ `dt` over the window, `corner_dist` = points × `Δs` |
| `<phase>_time`, `_dist` | Σ `dt` and points × `Δs` where the phase applies |
| `<phase>_pct_time`, `_pct_dist` | 100 × phase time (distance) / corner time (distance) |
| `brake_total_time` | braking + trail braking time, and its % of corner time |
| coasting before the brake point | when the entry is braking or trail braking, coasting from the last throttle-on point after the previous apex to `s_entry` is added to the coasting time and distance |
| `peak_lat_g` | max |lateral acceleration| in the window |
| `max_exit_throttle` | max throttle from `a` to `s_exit` |
| `steer_corrections` | number of sign changes of the steering rate in the window, ignoring rates inside the deadband |
| `exit_slip_pct` | 100 × (mean rear wheel speed / mean front wheel speed − 1) from `a` to `s_exit`, only where the front wheels turn faster than 1 rad/s |

## The app

Four pages. The sidebar reminds that driver 1 = stint 1, driver 2 = stint 2, and that the reference is the fastest lap of the session.

**Session overview.** The reference lap, number of comparable laps, stints and pit stop at a glance. Lap times vs the reference (warm-up lap hatched), the lap table with every flag, the excluded laps, the data-quality log and the known limitations.

**Lap comparison.** Pick any comparable lap against the reference:
- the track map coloured by time gained or lost per metre
- time lost per corner
- cumulative delta along the lap with corner zones shaded
- overlaid traces vs distance (speed, throttle, brake, steering, longitudinal and lateral g), with the corner phases of each lap as a strip and coasting dashed

**Corner deep-dive.** Pick a corner group: the four picked laps overlaid (speed, throttle, brake, steering vs metres from the window start) with entry and apex markers and exit speed, the metrics table, and the write-up.

**Opportunities.** Time at stake per group and per driver, sorted by size, with the recommendations for each group, the link between the last corner (C27) and C1 across the line, and an overall summary per driver.

## Assumptions

- **Two drivers, same car.** Stint 1 and stint 2 are driven by different drivers with the same car and setup (confirmed by the team), so differences reflect driving style.
- **The three ~15 s stops early in stint 1 are a blue flag** (pulling over to let faster traffic pass): the car slows from 3–6 m/s, stops smoothly with light braking, moves 2–5 m to the left before stopping and back after, at three different places, with nothing unusual in the other signals.
- **A standstill of at least 60 s is a pit stop.**
- **The car only crosses the start/finish line driving forward.**
- **Corners are a property of the track**, the same for every lap, taken from the centerline.
- **The apex is the point of maximum curvature** of the centerline, not the minimum speed point (minimum speed is measured separately as a metric).
- **The fastest comparable lap is the reference.** Lap 1 counts as warm-up because of cold tyres.
- **The estimator outputs are trusted** for position, speed and `s`, and GPS is not used directly.
- **Offsets measured at standstill hold for the whole session.**

## Limitations

- **Corner losses don't add up to the lap delta**, because the exits and straights between corner windows are not counted.
- **Logger gaps where `s` stalls** make the time at a given `s` locally wrong. They are not caught automatically: they were found with the speed-based time check and excluded by hand. Two picks still differ from that check and were kept: the fastest lap on the snail (lap 23) and lap 7 on C24.
- **`s` around short estimator NaN runs** can be off by ~2 m for a fraction of a second (the total distance is right). It only matters if a NaN run sits on a braking point.
- **Exit slip is a rough indicator.** It depends on the rolling radii and on inner vs outer wheels, and it is very sensitive to the length of the exit window: on lap 10, C24 went from ~42% to 67.7% when the exit point moved. At low speed over short windows, read it as a trend only.
- **`pickup_from_apex` is NaN** when the throttle never went off or is still off at the exit point.
- **Single-sample wheel speed glitches** (physically impossible jumps) are smeared into a small bump by the low-pass filter rather than removed.
- **Laps where `s` goes backwards** (a spin) are not handled. There are none on the grid in this session.
- **A corner window crossing the start/finish line** would stitch a lap's end to its own start. It doesn't happen in this session.
- **Manual exclusions and the excluded/warm-up laps are specific to this session** (set in `config.yaml`). A new session would need them checked again.
- **Outputs are rebuilt only when missing**, so changed raw data needs the processed files and the cache cleared by hand.
