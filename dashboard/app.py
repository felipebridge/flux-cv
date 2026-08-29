from __future__ import annotations

from pathlib import Path

import altair as alt
import streamlit as st
from data_loader import load_dashboard_data

_LEVEL_COLORS = {"LOW": "#28a745", "MODERATE": "#fd7e14", "HIGH": "#dc3545"}

st.set_page_config(page_title="FLUX Computer Vision", layout="wide")
st.title("FLUX Computer Vision")
st.caption("Vehicle count, pedestrian count, and traffic congestion level from a single video.")

output_dir = Path(st.sidebar.text_input("Output directory", value="outputs"))
data = load_dashboard_data(output_dir)

if data.tracks.empty:
    st.warning(
        f"No pipeline outputs found under `{output_dir}`. Run "
        "`python -m traffic_intelligence run --input data/raw/<video>.mp4` first."
    )
    st.stop()

summary = data.summary
traffic_level = summary.get("traffic_level", "N/A")
level_color = _LEVEL_COLORS.get(traffic_level, "#6c757d")

columns = st.columns(3)
columns[0].metric("Vehicles", summary.get("total_vehicles", 0))
columns[1].metric("Pedestrians", summary.get("total_pedestrians", 0))
with columns[2]:
    st.markdown("**Traffic level**")
    st.markdown(
        f"<span style='font-size:2rem;font-weight:700;color:{level_color}'>{traffic_level}</span>",
        unsafe_allow_html=True,
    )

st.subheader("Vehicles by class")
vehicles = data.tracks[data.tracks["class_name"] != "person"]
if not vehicles.empty:
    class_counts = vehicles["class_name"].value_counts().reset_index()
    class_counts.columns = ["class_name", "count"]
    st.altair_chart(
        alt.Chart(class_counts).mark_bar().encode(x="class_name", y="count", tooltip=["class_name", "count"]),
        use_container_width=True,
    )
else:
    st.info("No vehicles detected in this run.")

st.subheader("Annotated video")
if data.annotated_video_path is not None:
    st.video(str(data.annotated_video_path))
else:
    st.info("No annotated video found for this output directory.")

with st.expander("Raw track data"):
    st.dataframe(data.tracks, use_container_width=True)
