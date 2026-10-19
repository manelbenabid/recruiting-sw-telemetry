"""Page 3: one corner group: picked laps overlaid, metrics table and the write-up"""

import streamlit as st

from data import load_app_data, read_content
from plots import plot_group
from src.analysis import group_metrics
from src.config import CORNER_GROUPS, MANUAL_EXCLUSIONS

d = load_app_data()
ref, picks, gt = d["ref"], d["picks"], d["gt"]

st.title("Corner deep-dive")

name = st.segmented_control("Corner group", list(CORNER_GROUPS), default=list(CORNER_GROUPS)[0],
                            format_func=lambda g: f"{g} ({', '.join(f'C{c}' for c in CORNER_GROUPS[g])})"
                            if len(CORNER_GROUPS[g]) > 1 else g)
if name is None:
    st.stop()

st.caption("Laps compared in each group: the reference lap, the fastest lap in the group window "
           "(if it beats the reference), and the slowest lap of each stint. "
           "Warm-up laps and laps with a NaN run of s in the window are left out.")

p = picks[picks["group"] == name][["role", "lap", "time", "delta"]]
left, right = st.columns([2, 3])
with left:
    st.dataframe(p, hide_index=True, width="stretch",
                 column_config={"time": st.column_config.NumberColumn("time in window (s)", format="%.2f"),
                                "delta": st.column_config.NumberColumn("vs ref (s)", format="%+.2f")})
with right:
    if name in MANUAL_EXCLUSIONS:
        st.warning(f"Left out of {name} by hand: laps {', '.join(map(str, MANUAL_EXCLUSIONS[name]))} "
                   "(the time inside this window is broken on these laps, see below).")
    st.markdown("**How to read the plot.** Each colour is one lap: yellow = reference, blue = stint 1, "
                "pink = stint 2, purple = fastest in this group. Dashed = coasting (both pedals off). "
                "Triangles = corner entry (braking point), diamond = exit speed at the window end. "
                "The x axis is metres from the window start, the same for every lap.")

st.plotly_chart(plot_group(name, d["grid"], picks, gt, d["entries"], CORNER_GROUPS, d["main_apex"],
                           d["length"], ref), width="stretch", theme=None)

st.subheader("Metrics")
table, apex_pos = group_metrics(name, d["cm"], picks, gt, d["grid"], CORNER_GROUPS, d["main_apex"], d["length"])
table.columns = [f"C{c} · lap {lap}" for c, lap in table.columns]
st.dataframe(table.style.format(precision=2, na_rep="NaN"), width="stretch",
             height=(len(table) + 1) * 35 + 3)
st.caption(" · ".join(f"C{c} apex at {x:.1f} m from the window start" for c, x in apex_pos.items())
           + ". Positions are metres from the apex (- = before the apex). Speeds in m/s, times in s, "
             "brake in bar, throttle 0-1, slip in %.")

st.subheader("Write-up")
text = read_content("corners", f"{name}.md")
st.markdown(text if text else f"corners/{name}.md is missing from the content folder.")
