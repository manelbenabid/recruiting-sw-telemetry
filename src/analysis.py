import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.config import  (EXCLUDED_LAPS, WARMUP_LAPS, DISTANCE_GRID_STEP_M, 
                         DELTA_SMOOTH_M, DELTA_COLOR_CLIP_Q)



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


def _corner_spans(zones: pd.DataFrame, length: float):
    """
    
    returns (corner, start, end) for corner zones
    
    split in two if a zone crosses s = 0
    
    """
    spans = []
    for _, z in zones[zones["type"] == "corner"].iterrows():
        start, end = z["s_start"], z["s_end"]
        if end < start:  # wraps around the start/finish line
            spans += [(z["corner"], start, length), (z["corner"], 0, end)]
        else:
            spans.append((z["corner"], start, end))
    return spans


def plot_delta(d: pd.DataFrame, zones: pd.DataFrame, ref: int, lap: int,
               length: float):
    """
    delta vs distance
    corner zones are shaded so you can see if time loss happens inside corners 
    or on the straight after them
    a loss on a straight usually means a slower corner exit
    
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["s"], y=d["delta"], mode="lines",
                             name=f"lap {lap} vs lap {ref}", line=dict(width=2)))
    fig.add_hline(y=0, line=dict(color="grey", dash="dash"))

    for corner, start, end in _corner_spans(zones, length):
        fig.add_vrect(x0=start, x1=end, fillcolor="lightgrey", opacity=0.3,
                      line_width=0, layer="below")
        fig.add_annotation(x=(start + end) / 2, y=1, yref="paper", showarrow=False,
                           text=f"C{int(corner)}", font=dict(size=11))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="corner zone",
                             marker=dict(color="lightgrey", size=12, symbol="square")))

    fig.update_layout(title=f"Time delta: lap {lap} vs reference lap {ref} (+ = slower)",
                      xaxis_title="distance s (m)", yaxis_title="delta (s)",
                      width=1400, height=500, showlegend=True,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def plot_delta_map(d: pd.DataFrame, cl: pd.DataFrame, apexes: pd.DataFrame,
                   ref: int, lap: int, clip_q: float = DELTA_COLOR_CLIP_Q):
    """
    
    track (reference lap's line) coloured by where time is lost/gained
    red means losing, blue means gaining, white means equal (it's a gradient)
    might have small holes where position was NaN
    
    """
    length = cl.attrs["length"]
    x = np.interp(d["s"], cl["s"], cl["x"], period=length)   # centerline position at each grid s
    y = np.interp(d["s"], cl["s"], cl["y"], period=length)   # period: the loop closes at s = length
    lim = np.nanquantile(np.abs(d["loss_rate"]), clip_q)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        name=f"lap {lap} vs {ref}: red = losing, blue = gaining",
        marker=dict(size=5, color=d["loss_rate"], colorscale="RdBu_r",
                    cmin=-lim, cmax=lim, colorbar=dict(title="ms/m")),
        customdata=np.c_[d["s"], d["delta"], d["loss_rate"]],
        hovertemplate="s %{customdata[0]:.1f} m<br>delta %{customdata[1]:.3f} s"
                      "<br>rate %{customdata[2]:.2f} ms/m<extra></extra>"))

    main = apexes[apexes["is_main"]]
    fig.add_trace(go.Scatter(x=main["x"], y=main["y"], mode="text", name="corner",
                             text=[f"C{int(c)}" for c in main["corner"]],
                             textfont=dict(size=13)))
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name="start/finish",
                             marker=dict(symbol="x", size=14, color="black")))

    fig.update_layout(title=f"Where time goes: lap {lap} vs reference lap {ref}",
                      width=1400, height=850, showlegend=True,
                      legend=dict(orientation="h", y=-0.05),
                      xaxis_title="x (m)", yaxis_title="y (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig