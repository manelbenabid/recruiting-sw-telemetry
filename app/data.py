"""
Data for the app: everything the pages need, loaded once and cached.
Same steps as the first cell of notebooks/09_corner_study.ipynb, plus the corner times
of notebooks/08_analyze.ipynb. No plotting here (that is in plots.py).
"""

import pandas as pd
import streamlit as st

from src.segment import load_segmented
from src.loader import load_centerline
from src.config import CORNER_GROUPS, MANUAL_EXCLUSIONS, ISSUES_FILE, CONTENT_DIR
from src.analysis import (comparable_laps, reference_lap, time_delta, corner_entries,
                          corner_times, corner_metrics, group_times, pick_laps,
                          apply_exclusions)


@st.cache_data(show_spinner="Loading the session (the pipeline runs if its outputs are missing)...")
def load_app_data() -> dict:
    """
    loads the segmented session and runs the section 4 computations once.
    the pipeline (sections 2-3) runs inside load_segmented() if its outputs are missing.
    """
    data = load_segmented()
    grid, laps, zones, apexes, gaps = data["grid"], data["laps"], data["zones"], data["apexes"], data["gaps"]
    cl = load_centerline()
    length = float(cl.attrs["length"])       # kept apart: attrs are not relied on through the cache

    laps_cmp = comparable_laps(laps)
    laps_used = laps_cmp["lap"].tolist()
    ref = reference_lap(laps)

    entries = corner_entries(grid, apexes, zones, laps_used, length)
    ct = corner_times(grid, entries, zones, gaps, laps_used, ref, length)
    cm = corner_metrics(grid, entries, apexes, zones, laps_used, length)
    gt = group_times(grid, apexes, entries, zones, gaps, laps_used, CORNER_GROUPS, length)
    gt = apply_exclusions(gt, MANUAL_EXCLUSIONS)
    picks = pick_laps(gt, ref)

    return {"grid": grid, "laps": laps, "zones": zones, "apexes": apexes, "gaps": gaps,
            "stints": data["stints"], "standstills": data["standstills"],
            "cl": cl, "length": length, "laps_cmp": laps_cmp, "laps_used": laps_used, "ref": ref,
            "entries": entries, "ct": ct, "cm": cm, "gt": gt, "picks": picks,
            "main_apex": apexes[apexes["is_main"]].set_index("corner")["s"]}


@st.cache_data
def lap_delta(lap: int) -> pd.DataFrame:
    """time_delta of one lap vs the reference lap, cached per lap"""
    d = load_app_data()
    return time_delta(d["grid"], d["ref"], lap)


@st.cache_data
def read_issues() -> pd.DataFrame | None:
    """the data-quality log written during section 1, None if the file is missing"""
    return pd.read_csv(ISSUES_FILE) if ISSUES_FILE.exists() else None


def read_content(*parts: str) -> str | None:
    """text of a markdown file under the content folder, None if it is missing"""
    path = CONTENT_DIR.joinpath(*parts)
    return path.read_text(encoding="utf-8") if path.exists() else None
