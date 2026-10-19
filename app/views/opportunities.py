"""Page 4: what each driver should change per corner group, and how much time is at stake"""

import streamlit as st

from data import load_app_data, read_content
from plots import time_at_stake_bar
from src.config import CORNER_GROUPS

d = load_app_data()
ref, picks = d["ref"], d["picks"]

st.title("Opportunities")

# groups ranked by the larger of the two drivers' time at stake
slow = picks[picks["role"].str.startswith("slowest stint")]
order = slow.groupby("group")["delta"].max().sort_values(ascending=False).index.tolist()

st.plotly_chart(time_at_stake_bar(picks, order), width="stretch", theme=None)
st.caption(f"\"Up to\" = how much the slowest lap of that stint loses to the reference lap (lap {ref}) "
           "on the same group window. It is the most a driver could win back there, not a promise. "
           "The groups are timed separately and don't add up to a lap time.")

link = read_content("recommendations", "c27_to_c1.md")
if link:
    st.info("**C27 → C1**\n\n" + link, icon=":material/link:")


def role_delta(group: str, role: str) -> tuple[int, float]:
    """lap and delta vs ref of one pick of a group"""
    r = picks[(picks["group"] == group) & (picks["role"] == role)].iloc[0]
    return int(r["lap"]), float(r["delta"])


for g in order:
    corners = ", ".join(f"C{c}" for c in CORNER_GROUPS[g])
    with st.container(border=True):
        st.subheader(g if len(CORNER_GROUPS[g]) == 1 else f"{g} ({corners})")
        left, right = st.columns([1, 3])
        with left:
            for st_ in (1, 2):
                lap, delta = role_delta(g, f"slowest stint {st_}")
                st.metric(f"Driver {st_}: up to", f"{delta:.2f} s", f"slowest lap {lap} vs ref",
                          delta_color="off", delta_arrow="off")
            lap, delta = role_delta(g, "fastest")
            if lap != ref:
                st.metric("Fastest lap in this group", f"lap {lap}", f"{delta:+.2f} s vs ref",
                          delta_color="inverse")
        with right:
            text = read_content("recommendations", f"{g}.md")
            st.markdown(text if text else f"recommendations/{g}.md is missing from the content folder.")

st.subheader("Overall")
overall = read_content("recommendations", "overall.md")
st.markdown(overall if overall else "overall.md is missing from the content folder.")
