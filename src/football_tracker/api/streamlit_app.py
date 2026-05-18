"""Streamlit demo app. Implementation lands in Session 5."""

from __future__ import annotations

import streamlit as st

from football_tracker import __version__

st.set_page_config(
    page_title="Football Player Tracker",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Player Tracker")
st.caption(f"v{__version__} — detection + tracking demo")

st.markdown(
    """
    Upload an image or a short video clip and the model will detect and track
    players, the ball, referees and goalkeepers. The Streamlit UI and full
    inference path will be implemented in Session 5.
    """
)

st.info("Coming soon: image and video upload widgets, annotated preview and trajectory JSON download.")
