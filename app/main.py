"""
Entry point of the app. Run from the repo root:  streamlit run app/main.py
Sets the page, the logos and the navigation over the 4 pages in app/views/.
"""

from pathlib import Path

import streamlit as st

from src.config import SESSION_LABEL

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="E-agle telemetry", page_icon=str(ASSETS / "eagle.png"),
                   layout="wide")
st.logo(str(ASSETS / "logo_white.webp"), icon_image=str(ASSETS / "eagle.png"), size="large")

pages = [
    st.Page("views/session_overview.py", title="Session overview", icon=":material/flag:", default=True),
    st.Page("views/lap_comparison.py", title="Lap comparison", icon=":material/compare_arrows:"),
    st.Page("views/corner_deep_dive.py", title="Corner deep-dive", icon=":material/turn_sharp_right:"),
    st.Page("views/opportunities.py", title="Opportunities", icon=":material/trending_up:"),
]
nav = st.navigation(pages)

# the multiselect pills use the yellow primary colour as background, so their text has to be dark
st.html("<style>[data-testid='stMultiSelectTagsContainer'] [data-tag],"
        "[data-testid='stMultiSelectTagsContainer'] [data-tag] * { color: black !important; }</style>")

with st.sidebar:
    st.caption(SESSION_LABEL)
    st.caption("Driver 1 = stint 1, driver 2 = stint 2. The reference lap is the fastest lap of the session.")

nav.run()
