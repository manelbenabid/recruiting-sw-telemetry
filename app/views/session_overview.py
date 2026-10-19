"""Page 1: stints, pit stop, laps and data-quality issues"""

import pandas as pd
import streamlit as st

from data import load_app_data, read_issues, read_content
from plots import lap_times_bar
from src.config import WARMUP_LAPS, EXCLUDED_LAPS, MANUAL_EXCLUSIONS, SESSION_LABEL

d = load_app_data()
laps, laps_cmp, stints, standstills, ref = d["laps"], d["laps_cmp"], d["stints"], d["standstills"], d["ref"]

st.title("Session overview")
st.caption(SESSION_LABEL)

# pit stop = the pit standstill that ends where stint 2 starts
pit = standstills[(standstills["type"] == "pit")
                  & standstills["end_s"].round(3).isin(stints["start_s"].round(3))]
ref_time = float(laps.loc[laps["lap"] == ref, "lap_time"].iloc[0])
stint_laps = laps.groupby("stint")["lap"].agg(["min", "max"])

c = st.columns(4)
c[0].metric("Reference lap", f"lap {ref}", f"{ref_time:.2f} s", delta_color="off", delta_arrow="off")
c[1].metric("Comparable laps", len(laps_cmp), f"of {len(laps)} laps", delta_color="off", delta_arrow="off")
c[2].metric("Stints", len(stints), " / ".join(f"laps {r['min']}-{r['max']}" for _, r in stint_laps.iterrows()),
            delta_color="off", delta_arrow="off")
if len(pit):
    p = pit.iloc[0]
    c[3].metric("Pit stop", f"{p['duration_s']:.0f} s", f"{p['start_s']:.0f} s to {p['end_s']:.0f} s",
                delta_color="off", delta_arrow="off")

st.plotly_chart(lap_times_bar(laps_cmp, ref), width="stretch", theme=None)

left, right = st.columns([3, 2])
with left:
    st.subheader("Laps")
    table = laps[["lap", "stint", "lap_time", "has_pit", "has_stop", "has_nan_s"]].copy()
    table["warm_up"] = table["lap"].isin(WARMUP_LAPS)
    table["excluded"] = table["lap"].isin(EXCLUDED_LAPS)
    table["comparable"] = table["lap"].isin(d["laps_used"])
    table["delta_vs_ref"] = table["lap_time"] - ref_time
    table.insert(1, "", ["ref" if l == ref else "" for l in table["lap"]])
    st.dataframe(table, hide_index=True, width="stretch", height=600,
                 column_config={"lap_time": st.column_config.NumberColumn("lap time (s)", format="%.3f"),
                                "delta_vs_ref": st.column_config.NumberColumn("vs ref (s)", format="%+.3f"),
                                "has_nan_s": st.column_config.CheckboxColumn("NaN run of s")})
with right:
    st.subheader("What the flags mean")
    st.markdown(
        "- **has_pit**: the lap contains the pit stop, so it is not comparable.\n"
        "- **has_stop**: the car stood still during the lap.\n"
        "- **NaN run of s**: the position along the track (s) is missing for a while on this lap. "
        "The lap is kept, and the corners where it happens are flagged.\n"
        "- **warm-up**: kept for the lap overlays but left out of the corner picks.\n"
        "- **excluded**: left out of the analysis (set in config.yaml).\n"
        "- **comparable**: used in the comparisons."
    )
    excl = ", ".join(f"{g}: laps {', '.join(map(str, l))}" for g, l in MANUAL_EXCLUSIONS.items())
    st.info(f"Laps left out of single corner groups (time inside the corner is broken there): {excl}.")

st.subheader("Data-quality issues")
issues = read_issues()
if issues is None:
    st.warning("The issues log was not found (issues_file in config.yaml).")
else:
    st.dataframe(issues, hide_index=True, width="stretch")

with st.expander("Known limitations"):
    text = read_content("limitations.md")
    st.markdown(text if text else "limitations.md is missing from the content folder.")
