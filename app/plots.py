"""
All the Plotly figures of the app (plotting never goes in src/).
Ported from notebooks/08_analyze.ipynb and notebooks/09_corner_study.ipynb: same plots,
but the data is passed in instead of read from notebook globals, and they use the app's
yellow / black theme. The theme colours are read from .streamlit/config.toml.

Colours (checked for colour-blind separation on the black background):
- the reference lap is always the brand yellow
- stint 1 / driver 1 is blue and stint 2 / driver 2 is pink, everywhere
- in the corner study, the fastest lap is purple
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.segment import U_COL, THROTTLE_COL, BRAKE_COL, STEER_COL
from src.analysis import LAT_COL

STINT_COLORS = {1: "#1E9FD8", 2: "#E8577A"}
FASTEST_COLOR = "#8E6CF0"
PHASE_COLORS = {"braking": "#E0484E", "trail braking": "#8E6CF0", "lift": "#BF8A1A",
                "coasting": "#6B6B6B", "throttle on": "#1AA0B5"}
LOSS_COLOR, GAIN_COLOR = "#E0484E", "#1E9FD8"
MUTED = "#8A8A8A"
GRID_LINE = "rgba(255, 255, 255, 0.10)"
ZONE_FILL = "rgba(255, 255, 255, 0.07)"
COAST = "coasting"  # phase label for both pedals off
# delta map: diverging scale with a dark grey middle (no time difference) so it reads on black
DELTA_SCALE = [[0.0, "#4DB8FF"], [0.3, "#3D86BF"], [0.5, "#555555"], [0.7, "#C9474C"], [1.0, "#FF5A5F"]]
TRACK_LINE = "#1A1A1A"  # faint track outline under the delta map
LABEL_OFFSET_M = 14     # corner labels on the delta map sit this far outside the apex

LONG_COL = "imu_acceleration__x"
# every channel the overlays can show: column -> axis label
CHANNELS = {U_COL: "speed (m/s)", THROTTLE_COL: "throttle (0-1)",
            BRAKE_COL: "brake front (bar)", STEER_COL: "steering (deg, + = left)",
            LONG_COL: "long g (+ = accelerating)", LAT_COL: "lat g (+ = left)"}
DEFAULT_CHANNELS = [U_COL, THROTTLE_COL, BRAKE_COL, STEER_COL]


def theme() -> dict[str, str]:
    """theme colours from .streamlit/config.toml"""
    return {"primary": st.get_option("theme.primaryColor") or "gold",
            "bg": st.get_option("theme.backgroundColor") or "black",
            "text": st.get_option("theme.textColor") or "white"}


def style(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    """dark layout shared by every figure: title, then the legend under it, then the plot"""
    t = theme()
    title_px = 40 if title else 0  # room for the title at the top
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor=t["bg"],
                      plot_bgcolor=t["bg"], font=dict(color=t["text"]),
                      title=dict(text=title, font=dict(size=16), x=0, xanchor="left",
                                 yref="container", y=1, yanchor="top", pad=dict(t=12)) if title else None,
                      margin=dict(l=60, r=20, t=title_px + 60, b=40),
                      legend=dict(orientation="h", yref="container", y=1 - (title_px + 6) / height,
                                  yanchor="top", x=0),
                      hoverlabel=dict(bgcolor="#141414"))
    fig.update_xaxes(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE)
    fig.update_yaxes(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE)
    return fig


def lap_color(lap: int, ref: int, stint: int) -> str:
    """yellow for the reference lap, otherwise the colour of the lap's stint"""
    return theme()["primary"] if lap == ref else STINT_COLORS.get(stint, MUTED)


def _zone_spans(zones: pd.DataFrame, length: float) -> list[tuple[float, float, int]]:
    """corner zones as (x0, x1, corner), a zone crossing s = 0 is split in two"""
    spans = []
    for _, z in zones[zones["type"] == "corner"].iterrows():
        start, end, c = z["s_start"], z["s_end"], int(z["corner"])
        parts = [(start, length), (0.0, end)] if end < start else [(start, end)]
        spans += [(x0, x1, c) for x0, x1 in parts]
    return spans


def add_corner_zones(fig: go.Figure, zones: pd.DataFrame, length: float) -> go.Figure:
    """shades every corner zone and writes its corner number on top (all subplot rows)"""
    for x0, x1, c in _zone_spans(zones, length):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=ZONE_FILL, line_width=0, layer="below",
                      row="all", col=1)
        fig.add_annotation(x=(x0 + x1) / 2, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"C{c}", font=dict(size=10, color=MUTED))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="corner zone",
                             marker=dict(color="rgba(255, 255, 255, 0.25)", size=12, symbol="square")))
    return fig


## page 1: session overview

def lap_times_bar(laps_cmp: pd.DataFrame, ref: int) -> go.Figure:
    """lap time minus the reference lap time per comparable lap, coloured by stint"""
    ref_time = float(laps_cmp.loc[laps_cmp["lap"] == ref, "lap_time"].iloc[0])
    d = laps_cmp.sort_values("lap")
    fig = go.Figure()
    for st_, g in d.groupby("stint"):
        delta = g["lap_time"] - ref_time
        fig.add_trace(go.Bar(
            x=g["lap"], y=delta, name=f"stint {st_} (driver {st_})",
            marker=dict(color=[lap_color(l, ref, st_) for l in g["lap"]],
                        pattern_shape=["/" if w else "" for w in g["is_warmup"]],
                        cornerradius=4),
            customdata=np.c_[g["lap_time"], g["is_warmup"]],
            hovertemplate="lap %{x}<br>lap time %{customdata[0]:.3f} s<br>"
                          "+%{y:.3f} s vs ref<extra></extra>"))
    fig.add_annotation(x=ref, y=0, text=f"ref lap {ref}<br>{ref_time:.2f} s", showarrow=True,
                       arrowhead=0, ax=0, ay=-40, font=dict(color=theme()["primary"]))
    fig.update_layout(bargap=0.25, xaxis=dict(title="lap", dtick=1),
                      yaxis=dict(title="lap time - ref (s)", rangemode="tozero"))
    return style(fig, 360, "Lap times vs the reference lap (hatched = warm-up lap)")


## page 2: lap comparison

def delta_map(d: pd.DataFrame, cl: pd.DataFrame, apexes: pd.DataFrame, length: float,
              lap: int, ref: int, clip_q: float) -> go.Figure:
    """
    track map coloured by loss rate: red = losing time, blue = gaining (08 cell 7).
    the colour scale fades to dark grey at 0 so only real gains and losses stand out on black,
    and the corner labels sit just outside each corner so they don't cover the track.
    """
    x = np.interp(d["s"], cl["s"], cl["x"], period=length)
    y = np.interp(d["s"], cl["s"], cl["y"], period=length)
    lim = np.nanquantile(np.abs(d["loss_rate"]), clip_q)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cl["x"], y=cl["y"], mode="lines", line=dict(color=TRACK_LINE, width=14),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="red = losing, blue = gaining, grey = same",
        marker=dict(size=8, color=d["loss_rate"], colorscale=DELTA_SCALE, cmin=-lim, cmax=lim,
                    colorbar=dict(title="ms/m", thickness=12)),
        customdata=np.c_[d["s"], d["delta"], d["loss_rate"]],
        hovertemplate="s %{customdata[0]:.1f} m<br>delta %{customdata[1]:.3f} s"
                      "<br>rate %{customdata[2]:.2f} ms/m<extra></extra>"))

    # corner labels pushed outward from the apex: to the right of the driving direction for a
    # left corner, to the left for a right corner
    main = apexes[apexes["is_main"]]
    s = main["s"].to_numpy()
    dx = np.interp(s + 1, cl["s"], cl["x"], period=length) - np.interp(s - 1, cl["s"], cl["x"], period=length)
    dy = np.interp(s + 1, cl["s"], cl["y"], period=length) - np.interp(s - 1, cl["s"], cl["y"], period=length)
    norm = np.hypot(dx, dy)
    side = np.where(main["direction"].to_numpy() == "L", 1.0, -1.0) * LABEL_OFFSET_M
    lx = main["x"].to_numpy() + side * dy / norm
    ly = main["y"].to_numpy() - side * dx / norm
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="text", name="corner",
                             text=[f"C{int(c)}" for c in main["corner"]],
                             textfont=dict(size=11, color=MUTED), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name="start/finish",
                             marker=dict(symbol="x", size=13, color=theme()["primary"])))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False, visible=False)
    fig.update_xaxes(showgrid=False, visible=False)
    return style(fig, 620, f"Where time goes: lap {lap} vs ref lap {ref}")


def corner_loss_bar(ct: pd.DataFrame, lap: int, ref: int) -> go.Figure:
    """time lost (+) or gained (-) per corner vs the reference lap (08 cell 21)"""
    c = ct[ct["lap"] == lap].sort_values("corner")
    loss = c["loss"].fillna(0)
    bad = c[~c["valid"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[f"C{k}" for k in c["corner"]], x=loss, orientation="h", name="lost (+) / gained (-)",
        marker=dict(color=np.where(loss > 0, LOSS_COLOR, GAIN_COLOR), cornerradius=3),
        customdata=np.c_[c["entry_diff_m"], c["source_lap"], c["source_ref"]],
        hovertemplate="%{y}: %{x:+.3f} s<br>enters %{customdata[0]:+.1f} m vs ref"
                      "<br>lap entry: %{customdata[1]}<br>ref entry: %{customdata[2]}<extra></extra>"))
    fig.add_trace(go.Scatter(y=[f"C{k}" for k in bad["corner"]], x=[0] * len(bad), mode="markers",
                             name="not valid (edge inside a NaN run of s)",
                             marker=dict(symbol="x", size=10, color=MUTED)))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="loss (s)", zeroline=True, zerolinecolor=MUTED)
    return style(fig, 620, f"Time per corner: lap {lap} vs ref lap {ref}")


def delta_vs_distance(d: pd.DataFrame, zones: pd.DataFrame, length: float,
                      lap: int, ref: int, color: str) -> go.Figure:
    """cumulative time delta along the lap, corner zones shaded (08 cell 6)"""
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=d["s"], y=d["delta"], mode="lines", name=f"lap {lap} vs lap {ref}",
                             line=dict(width=2, color=color),
                             hovertemplate="s %{x:.1f} m<br>delta %{y:+.3f} s<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dash", width=1))
    add_corner_zones(fig, zones, length)
    fig.update_layout(xaxis_title="distance s (m)", yaxis_title="delta (s, + = slower)")
    return style(fig, 360, f"Time delta along the lap: lap {lap} vs ref lap {ref}")


def lap_overlay(grid: pd.DataFrame, zones: pd.DataFrame, length: float, laps: list[tuple[int, str, str]],
                channels: list[str]) -> go.Figure:
    """
    stacked channels vs distance for a few laps, coasting dashed (as in plot_group),
    corner zones shaded, and a phase strip at the bottom (one band per lap).
    laps = [(lap, label, colour)]
    """
    n = len(channels)
    fig = make_subplots(rows=n + 1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[1.0] * n + [0.18 * len(laps)])
    for lap, label, color in laps:
        lg = grid[grid["lap"] == lap].sort_values("s")
        coast = (lg["phase"] == COAST).to_numpy()
        coast_ext = coast | np.roll(coast, 1) | np.roll(coast, -1)  # overlap one point so lines join
        for i, col in enumerate(channels, start=1):
            y = lg[col].to_numpy()
            fig.add_trace(go.Scatter(x=lg["s"], y=np.where(coast, np.nan, y), mode="lines",
                                     line=dict(color=color, width=1.6), name=label, legendgroup=label,
                                     showlegend=(i == 1),
                                     hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"),
                          row=i, col=1)
            fig.add_trace(go.Scatter(x=lg["s"], y=np.where(coast_ext, y, np.nan), mode="lines",
                                     line=dict(color=color, width=1.6, dash="dash"),
                                     name=f"{label} coasting", legendgroup=label, showlegend=False,
                                     hoverinfo="skip"), row=i, col=1)

        # phase strip
        for p, pc in PHASE_COLORS.items():
            m = (lg["phase"] == p).to_numpy()
            fig.add_trace(go.Scatter(x=lg["s"].to_numpy()[m], y=[label] * int(m.sum()), mode="markers",
                                     marker=dict(symbol="square", size=9, color=pc), name=p,
                                     legendgroup=f"phase {p}", showlegend=(lap == laps[0][0]),
                                     hovertemplate=f"{label}: {p}<extra></extra>"),
                          row=n + 1, col=1)

    for i, col in enumerate(channels, start=1):
        fig.update_yaxes(title_text=CHANNELS[col], title_font=dict(size=11), row=i, col=1)
    fig.update_yaxes(showgrid=False, row=n + 1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="coasting (dashed)",
                             line=dict(color=MUTED, dash="dash")), row=1, col=1)
    add_corner_zones(fig, zones, length)
    fig.update_xaxes(title_text="distance s (m)", row=n + 1, col=1)
    fig.update_layout(hovermode="x unified")
    return style(fig, 190 * n + 70 * len(laps) + 120, "Overlays and phases vs distance")


## page 3: corner deep-dive

def plot_group(name: str, grid: pd.DataFrame, picks: pd.DataFrame, gt: pd.DataFrame,
               entries: pd.DataFrame, groups: dict[str, list[int]], main_apex: pd.Series,
               length: float, ref: int) -> go.Figure:
    """picked laps of one group, 4 stacked channels vs metres from window start (09 cell 4)"""
    channels = DEFAULT_CHANNELS
    p = picks[picks["group"] == name]
    roles = p.groupby("lap", sort=False)["role"].agg(" + ".join)  # ref + fastest merged if same lap
    w = gt[gt["group"] == name].iloc[0]
    s0, s1 = float(w["s_start"]), float(w["s_end"])
    span = np.mod(s1 - s0, length)
    corners = groups[name]
    stint = gt.set_index(["group", "lap"])["stint"]

    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.03)
    for lap, role in roles.items():
        lg = grid[grid["lap"] == lap].copy()
        lg["d"] = np.mod(lg["s"] - s0, length)  # metres from window start
        lg = lg[lg["d"] <= span].sort_values("d").reset_index(drop=True)
        color = (FASTEST_COLOR if role == "fastest" and lap != ref
                 else lap_color(lap, ref, int(stint[(name, lap)])))
        label = f"lap {lap} ({role})"

        coast = (lg["phase"] == COAST).to_numpy()
        coast_ext = coast | np.roll(coast, 1) | np.roll(coast, -1)  # overlap one point so lines join
        lap_entries = entries[(entries["lap"] == lap) & (entries["corner"].isin(corners))]

        for i, col in enumerate(channels, start=1):
            y = lg[col].to_numpy()
            fig.add_trace(go.Scatter(x=lg["d"], y=np.where(coast, np.nan, y), mode="lines",
                                     line=dict(color=color, width=2), name=label, legendgroup=str(lap),
                                     showlegend=(i == 1),
                                     hovertemplate=f"lap {lap}: %{{y:.2f}}<extra></extra>"),
                          row=i, col=1)
            fig.add_trace(go.Scatter(x=lg["d"], y=np.where(coast_ext, y, np.nan), mode="lines",
                                     line=dict(color=color, width=2, dash="dash"),
                                     name=f"{label} coasting", legendgroup=str(lap), showlegend=False,
                                     hoverinfo="skip"), row=i, col=1)

            for _, e in lap_entries.iterrows():  # braking point / corner entry
                xe = float(np.mod(e["s_entry"] - s0, length))
                fig.add_trace(go.Scatter(x=[xe], y=[np.interp(xe, lg["d"], y)], mode="markers",
                                         marker=dict(color=color, size=11, symbol="triangle-down",
                                                     line=dict(color="black", width=1)),
                                         name=label, legendgroup=str(lap), showlegend=False,
                                         hovertext=f"lap {lap}: C{int(e['corner'])} entry ({e['entry_source']})",
                                         hoverinfo="text"), row=i, col=1)
            fig.update_yaxes(title_text=CHANNELS[col], title_font=dict(size=11), row=i, col=1)

        u_exit = float(lg[U_COL].iloc[-1])  # exit speed at window end
        fig.add_trace(go.Scatter(x=[float(lg["d"].iloc[-1])], y=[u_exit], mode="markers",
                                 marker=dict(color=color, size=12, symbol="diamond",
                                             line=dict(color="black", width=1)),
                                 name=label, legendgroup=str(lap), showlegend=False,
                                 hovertext=f"lap {lap}: exit {u_exit:.2f} m/s", hoverinfo="text"),
                      row=1, col=1)

    for c in corners:  # main apexes of the group
        x = float(np.mod(main_apex[c] - s0, length))
        fig.add_vline(x=x, line_dash="dot", line_color=MUTED)
        fig.add_annotation(x=x, y=1.0, yref="paper", text=f"C{c} apex", showarrow=False,
                           yanchor="bottom", font=dict(color=MUTED))

    # legend keys for the symbols
    for sym, txt in [("triangle-down", "entry / braking point"), ("diamond", "exit speed")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=txt,
                                 marker=dict(color=MUTED, size=11, symbol=sym)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="coasting",
                             line=dict(color=MUTED, dash="dash")), row=1, col=1)

    fig.update_xaxes(title_text="metres from window start", row=len(channels), col=1)
    return style(fig, 950, f"{name}: picked laps")


## page 4: opportunities

def time_at_stake_bar(picks: pd.DataFrame, order: list[str]) -> go.Figure:
    """
    per group, how much the slowest lap of each stint loses to the reference lap
    on the group window: the "up to" time each driver can win back there
    """
    fig = go.Figure()
    for st_ in (1, 2):
        p = picks[picks["role"] == f"slowest stint {st_}"].set_index("group").reindex(order)
        fig.add_trace(go.Bar(
            y=order, x=p["delta"], orientation="h", name=f"driver {st_} (stint {st_})",
            marker=dict(color=STINT_COLORS[st_], cornerradius=4),
            text=[f"{v:.2f} s" for v in p["delta"]], textposition="outside",
            textfont=dict(color=theme()["text"]),
            customdata=p["lap"],
            hovertemplate="%{y}, driver " + str(st_) + ": up to %{x:.2f} s"
                          "<br>(lap %{customdata} vs ref)<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="time at stake vs ref lap (s)", rangemode="tozero")
    return style(fig, 420, "Time at stake per corner group")
