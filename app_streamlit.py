# app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

DATA_DIR = Path("data")
st.set_page_config(layout="wide", page_title="FPL Lineup Explorer")

st.title("FPL Lineup Explorer — Predicted Lineups & Player Analysis")

pred_path = DATA_DIR / "predictions_next_gw.csv"
sel_path = DATA_DIR / "selected_squad.csv"
if not pred_path.exists():
    st.error("predictions_next_gw.csv not found in data/. Run the pipeline first (run_fpl_pipeline.py).")
    st.stop()

# Load predictions and selected squad
pred_df = pd.read_csv(pred_path)
# Ensure required columns exist
required_pred_cols = {"player_id", "name", "team", "position", "value", "predicted_points"}
if not required_pred_cols.issubset(set(pred_df.columns)):
    st.error(f"predictions_next_gw.csv missing required columns: {required_pred_cols - set(pred_df.columns)}")
    st.stop()

# Normalize types for numeric plotting
pred_df["value"] = pd.to_numeric(pred_df["value"], errors="coerce")
pred_df["predicted_points"] = pd.to_numeric(pred_df["predicted_points"], errors="coerce")
# drop entries without numeric value/points
pred_df = pred_df.dropna(subset=["value", "predicted_points"]).reset_index(drop=True)
pred_df["position_name"] = pred_df["position"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})

if sel_path.exists():
    sel_df = pd.read_csv(sel_path)
    sel_df["value"] = pd.to_numeric(sel_df["value"], errors="coerce")
    sel_df["predicted_points"] = pd.to_numeric(sel_df["predicted_points"], errors="coerce")
    sel_df = sel_df.dropna(subset=["value", "predicted_points"]).reset_index(drop=True)
    sel_df["position_name"] = sel_df["position"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
else:
    sel_df = pd.DataFrame(columns=list(pred_df.columns) + ["position_name"])

# Top metrics
st.header("Model Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Players", len(pred_df))
col2.metric("Selected Squad", len(sel_df) if not sel_df.empty else 0)
if not sel_df.empty:
    col3.metric("Predicted Squad Points", round(sel_df["predicted_points"].sum(), 2))
else:
    col3.metric("Predicted Squad Points", "—")

# Sidebar filters
st.sidebar.header("Filters")
positions = sorted(pred_df["position_name"].unique())
selected_positions = st.sidebar.multiselect("Position", options=positions, default=positions)
selected_teams = st.sidebar.multiselect("Team", options=sorted(pred_df["team"].unique()), default=[])
min_pp = st.sidebar.slider(
    "Min predicted points",
    float(pred_df["predicted_points"].min()),
    float(pred_df["predicted_points"].max()),
    float(pred_df["predicted_points"].min()),
)

# Apply filters
df_filtered = pred_df[pred_df["position_name"].isin(selected_positions)]
if selected_teams:
    df_filtered = df_filtered[df_filtered["team"].isin(selected_teams)]
df_filtered = df_filtered[df_filtered["predicted_points"] >= min_pp]

st.subheader("All Players (filtered)")
st.dataframe(
    df_filtered.sort_values("predicted_points", ascending=False).reset_index(drop=True),
    use_container_width=True,
)

st.subheader("Model-selected 15-player Squad")
if sel_df.empty:
    st.info("No selected_squad.csv found. Run the pipeline to generate it.")
else:
    st.dataframe(
        sel_df[["name", "team", "position_name", "value", "predicted_points"]]
        .sort_values("predicted_points", ascending=False),
        use_container_width=True,
    )
    fig = px.pie(sel_df, names="position_name", title="Position Breakdown")
    st.plotly_chart(fig, use_container_width=True)

    # Starting XI suggestion
    st.subheader("Suggested Starting XI (auto)")

    def suggest_starting_xi(squad_df):
        if squad_df.empty:
            return squad_df
        gk = squad_df[squad_df["position_name"] == "GK"].nlargest(1, "predicted_points")
        # if no GK present, return empty
        if gk.empty:
            return pd.DataFrame()
        formations = [(3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (5, 3, 2)]
        for d, m, f in formations:
            pick_defs = squad_df[squad_df["position_name"] == "DEF"].nlargest(d, "predicted_points")
            pick_mids = squad_df[squad_df["position_name"] == "MID"].nlargest(m, "predicted_points")
            pick_fwds = squad_df[squad_df["position_name"] == "FWD"].nlargest(f, "predicted_points")
            if len(pick_defs) == d and len(pick_mids) == m and len(pick_fwds) == f:
                starters = pd.concat([gk, pick_defs, pick_mids, pick_fwds])
                return starters.sort_values("predicted_points", ascending=False)
        # fallback: top 10 non-gk + gk
        non_gk = squad_df[squad_df["position_name"] != "GK"].nlargest(10, "predicted_points")
        starters = pd.concat([gk, non_gk])
        return starters.sort_values("predicted_points", ascending=False)

    starters = suggest_starting_xi(sel_df)
    if not starters.empty:
        st.table(starters[["name", "team", "position_name", "predicted_points"]])
        captain = starters.nlargest(1, "predicted_points")
        if not captain.empty:
            st.markdown(
                f"**Recommended Captain:** {captain.iloc[0]['name']} — predicted {captain.iloc[0]['predicted_points']:.2f}"
            )
    else:
        st.info("Unable to suggest a starting XI from the selected squad (missing players).")

# Value plot (fixed / robust) - Replace the existing section with this
st.subheader("Value Picks & Distribution")

# Create a copy and do robust filtering
plot_df = pred_df.copy()

# Debug info
st.write(f"Total players before filtering: {len(plot_df)}")

# Filter out invalid data
plot_df = plot_df[
    (plot_df["value"] > 0) & 
    (plot_df["predicted_points"].notna()) & 
    (plot_df["value"].notna())
].copy()

st.write(f"Players after filtering (value > 0 and valid data): {len(plot_df)}")

if plot_df.empty:
    st.warning("No valid data available for distribution plot. Check your predictions_next_gw.csv file.")
else:
    # Calculate points per value
    plot_df["points_per_value"] = plot_df["predicted_points"] / plot_df["value"]
    
    # Show value ranges for debugging
    st.write(f"Value range: {plot_df['value'].min():.2f} - {plot_df['value'].max():.2f}")
    st.write(f"Predicted points range: {plot_df['predicted_points'].min():.2f} - {plot_df['predicted_points'].max():.2f}")
    
    # Create scatter plot
    fig2 = px.scatter(
        plot_df,
        x="value",
        y="predicted_points",
        color="position_name",
        hover_data=["name", "team", "points_per_value"],
        title="Value Picks & Distribution",
        template="plotly_dark",
    )
    
    # Make points more visible
    fig2.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color='white')))
    fig2.update_layout(
        xaxis_title="Player Value (M)", 
        yaxis_title="Predicted Points",
        height=600
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Show top value picks
    st.subheader("Top 10 Value Picks (Points per £M)")
    top_value = plot_df.nlargest(10, "points_per_value")[
        ["name", "team", "position_name", "value", "predicted_points", "points_per_value"]
    ]
    top_value["points_per_value"] = top_value["points_per_value"].round(2)
    st.dataframe(top_value, use_container_width=True)
st.markdown("Download current predictions or selected squad:")
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download predictions CSV", data=pred_df.to_csv(index=False), file_name="predictions_next_gw.csv")
with c2:
    if not sel_df.empty:
        st.download_button("Download selected squad CSV", data=sel_df.to_csv(index=False), file_name="selected_squad.csv")
