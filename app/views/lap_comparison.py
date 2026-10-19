"""Page 2: one lap vs the reference lap: delta map, time per corner, delta along the lap, overlays"""

import streamlit as st

from data import load_app_data, lap_delta
from plots import (delta_map, corner_loss_bar, delta_vs_distance, lap_overlay, lap_color,
                   CHANNELS, DEFAULT_CHANNELS)
from src.config import DELTA_COLOR_CLIP_Q

d = load_app_data()
ref, laps_cmp, grid = d["ref"], d["laps_cmp"], d["grid"]

st.title("Lap comparison")

others = laps_cmp[laps_cmp["lap"] != ref].sort_values("lap")
slowest = int(laps_cmp[~laps_cmp["is_warmup"]].iloc[-1]["lap"])  # sorted fastest to slowest, warm-up skipped
labels = {int(r["lap"]): f"lap {int(r['lap'])}  (stint {int(r['stint'])}, {r['lap_time']:.2f} s)"
          for _, r in others.iterrows()}

f1, f2 = st.columns([1, 2])
lap = f1.selectbox("Lap", list(labels), index=list(labels).index(slowest), format_func=labels.get)
channels = f2.multiselect("Channels", list(CHANNELS), default=DEFAULT_CHANNELS, format_func=CHANNELS.get)

stint = int(others.loc[others["lap"] == lap, "stint"].iloc[0])
color = lap_color(lap, ref, stint)
ref_time = float(laps_cmp.loc[laps_cmp["lap"] == ref, "lap_time"].iloc[0])
lap_time = float(laps_cmp.loc[laps_cmp["lap"] == lap, "lap_time"].iloc[0])

c = st.columns(3)
c[0].metric("Reference lap", f"lap {ref}", f"{ref_time:.2f} s", delta_color="off", delta_arrow="off")
c[1].metric(f"Lap {lap}", f"{lap_time:.2f} s", f"stint {stint} (driver {stint})", delta_color="off", delta_arrow="off")
c[2].metric("Lap time vs ref", f"{lap_time - ref_time:+.2f} s")

delta = lap_delta(lap)
left, right = st.columns([3, 2])
left.plotly_chart(delta_map(delta, d["cl"], d["apexes"], d["length"], lap, ref, DELTA_COLOR_CLIP_Q),
                  width="stretch", theme=None)
right.plotly_chart(corner_loss_bar(d["ct"], lap, ref), width="stretch", theme=None)
st.caption("Each corner is timed from the earlier of the two laps' entries to the corner zone end. "
           "The straights between corners are not counted, so the corner losses don't add up to the lap delta.")

st.plotly_chart(delta_vs_distance(delta, d["zones"], d["length"], lap, ref, color), width="stretch", theme=None)

if channels:
    channels = [col for col in CHANNELS if col in channels]  # keep a fixed panel order
    laps = [(ref, f"lap {ref} (ref)", lap_color(ref, ref, 1)), (lap, f"lap {lap}", color)]
    st.plotly_chart(lap_overlay(grid, d["zones"], d["length"], laps, channels), width="stretch", theme=None)
    st.caption("Drag on a chart to zoom into a corner, double-click to zoom out. "
               "Dashed = coasting (both pedals off).")
else:
    st.info("Pick at least one channel to see the overlays.")
