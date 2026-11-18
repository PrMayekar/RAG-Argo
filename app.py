# NETCDF-FILES/app.py

# ==========================================================
# 🌊 ARGO PostgreSQL 10-Algorithm Dashboard (Full Version)
# ==========================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="ARGO 10-Algorithm Dashboard", layout="wide")
st.title("🌊 ARGO PostgreSQL 10-Algorithm Demo Dashboard")

# ==========================================================
# ⚙️ Connect to PostgreSQL
# ==========================================================
# Update username/password/dbname as needed
engine = create_engine("postgresql://postgres:password@localhost:5432/argo_db")

# ==========================================================
# ⚡ Select Float from Database (WMO ID only)
# ==========================================================
floats = pd.read_sql("SELECT id, wmo_id, dac, institution, platform_type FROM argo_floats ORDER BY id", engine)
float_options = [str(row['wmo_id']) for _, row in floats.iterrows()]
selected_float = st.selectbox("Select Float (WMO ID)", float_options)
float_db_id = floats.loc[floats['wmo_id']==selected_float, 'id'].values[0]

# ==========================================================
# ⚡ Load Profiles & Measurements
# ==========================================================
query = f"""
SELECT m.*, p.latitude, p.longitude, p.cycle_number, p.profile_date
FROM measurements m
JOIN profiles p ON m.profile_id = p.id
WHERE p.float_id = {float_db_id}
ORDER BY p.cycle_number, m.depth_level;
"""
df = pd.read_sql(query, engine)

if df.empty:
    st.error("No measurement data found for this float!")
    st.stop()

# Drop NaNs for ML processing
df = df.dropna(subset=["temperature","pressure","latitude","longitude"])
st.write("Data Preview:", df.head())

# ==========================================================
# 1️⃣ Trajectory + Past & Future Prediction
# ==========================================================
st.subheader("1️⃣ Trajectory + Past & Future Prediction")
df['profile_date'] = pd.to_datetime(df['profile_date'])
df_traj = df.groupby('profile_date')[['latitude','longitude']].mean().reset_index()
df_traj = df_traj.dropna()

if len(df_traj) >= 2:
    t0 = df_traj['profile_date'].min()
    df_traj['days'] = (df_traj['profile_date'] - t0).dt.total_seconds() / 86400

    X = df_traj['days'].values.reshape(-1,1)
    lat_model = LinearRegression().fit(X, df_traj['latitude'])
    lon_model = LinearRegression().fit(X, df_traj['longitude'])

    past_days = np.linspace(df_traj['days'].min()-5, df_traj['days'].min()-1, 5).reshape(-1,1)
    future_days = np.linspace(df_traj['days'].max()+1, df_traj['days'].max()+10, 10).reshape(-1,1)

    past_pred = pd.DataFrame({
        "latitude": lat_model.predict(past_days),
        "longitude": lon_model.predict(past_days)
    })
    future_pred = pd.DataFrame({
        "latitude": lat_model.predict(future_days),
        "longitude": lon_model.predict(future_days)
    })
    observed = df_traj[['latitude','longitude']]

    fig = go.Figure()

    # Past line (blue)
    fig.add_trace(go.Scattermapbox(
        lat=past_pred["latitude"], lon=past_pred["longitude"],
        mode="lines+markers", line=dict(color="blue", width=3),
        name="Past"
    ))

    # Observed line (black)
    fig.add_trace(go.Scattermapbox(
        lat=observed["latitude"], lon=observed["longitude"],
        mode="lines+markers", line=dict(color="black", width=3),
        name="Observed"
    ))

    # Future line (red)
    fig.add_trace(go.Scattermapbox(
        lat=future_pred["latitude"], lon=future_pred["longitude"],
        mode="lines+markers", line=dict(color="red", width=3),
        name="Future"
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=3,
        mapbox_center={"lat": df_traj["latitude"].mean(), "lon": df_traj["longitude"].mean()},
        height=600,  # Bigger map
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, width='stretch')
else:
    st.warning("Not enough trajectory data for prediction.")

# ==========================================================
# 2️⃣ T-S Diagram
# ==========================================================
st.subheader("2️⃣ T-S Diagram")
fig = px.scatter(df, x="temperature", y="pressure", color="cycle_number")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, width='stretch')

# ==========================================================
# 3️⃣ Temperature Profile Fit
# ==========================================================
st.subheader("3️⃣ Temperature Profile Fit")
df_fit = df[["pressure","temperature"]].dropna()
if len(df_fit) >= 2:
    X = df_fit[["pressure"]]
    y = df_fit["temperature"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(np.linspace(X["pressure"].min(), X["pressure"].max(), 50).reshape(-1,1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=X["pressure"], name="Actual"))
    fig.add_trace(go.Scatter(x=pred, y=np.linspace(X["pressure"].min(), X["pressure"].max(), 50),
                         name="Predicted", line=dict(color="red", dash="dash")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width='stretch')
else:
    st.warning("Not enough data to fit Temperature Profile")

# ==========================================================
# 4️⃣ K-Means Clustering
# ==========================================================
st.subheader("4️⃣ K-Means Clustering")
k = st.slider("Number of clusters", 2, 5, 3)
km = KMeans(n_clusters=k, n_init=10)
df["cluster"] = km.fit_predict(df[["latitude","longitude"]])
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="cluster")
fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=2, height=400)
st.plotly_chart(fig, width='stretch')

# ==========================================================
# 5️⃣ PCA Projection
# ==========================================================
st.subheader("5️⃣ PCA Projection")
X_scaled = StandardScaler().fit_transform(df[["latitude","longitude"]])
pca = PCA(n_components=2)
comp = pca.fit_transform(X_scaled)
fig = px.scatter(x=comp[:,0], y=comp[:,1])
st.plotly_chart(fig, width='stretch')

# ==========================================================
# 6️⃣ Heatmap
# ==========================================================
st.subheader("6️⃣ Heatmap")
df["lat_bin"] = df["latitude"].round()
df["lon_bin"] = df["longitude"].round()
df_heat = df.groupby(["lat_bin","lon_bin"]).size().reset_index(name="count")
fig = px.density_mapbox(df_heat, lat="lat_bin", lon="lon_bin", z="count", radius=10)
fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=2, height=400)
st.plotly_chart(fig, width='stretch')

# ==========================================================
# 7️⃣ kNN Nearest Neighbors
# ==========================================================
st.subheader("7️⃣ kNN Nearest Neighbors")
lat_input = st.number_input("Enter Latitude", value=float(df["latitude"].mean()))
lon_input = st.number_input("Enter Longitude", value=float(df["longitude"].mean()))
nn = NearestNeighbors(n_neighbors=3)
nn.fit(df[["latitude","longitude"]])
dist, idx = nn.kneighbors([[lat_input, lon_input]])
st.write("Nearest Points:")
st.dataframe(df.iloc[idx[0]].reset_index(drop=True))

# ==========================================================
# 8️⃣ Float Summary Table
# ==========================================================
st.subheader("8️⃣ Float Summary Table")
summary = df.groupby("cycle_number")[["temperature","pressure"]].agg(["min","max","mean"]).reset_index()
st.dataframe(summary)

# ==========================================================
# 9️⃣ Temperature Statistics
# ==========================================================
st.subheader("9️⃣ Temperature Statistics")
st.write(df["temperature"].describe())

# ==========================================================
# 🔟 Top 3 Profiles Line Charts
# ==========================================================
st.subheader("🔟 Top 3 Profiles Line Charts")
top_cycles = df["cycle_number"].drop_duplicates().sort_values()[:3]
for cycle in top_cycles:
    profile_df = df[df["cycle_number"] == cycle]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=profile_df["temperature"], y=profile_df["pressure"],
                             mode="lines+markers", name=f"Profile {cycle}"))
    fig.update_yaxes(autorange="reversed", title="Pressure (dbar)")
    fig.update_xaxes(title="Temperature (°C)")
    st.plotly_chart(fig, width='stretch')

# ── paste this at the very bottom of app.py ──────────────────

from query import ask   # imports the ask() function above

st.divider()
st.subheader("💬 Ask the Ocean AI")
st.caption("Ask anything about the Argo float data — powered by Groq LLaMA-3.3 70B")

# Keep chat history across reruns using session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show existing messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("e.g. What was the temperature at cycle 5?"):

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get AI answer
    with st.chat_message("assistant"):
        with st.spinner("Searching ocean data..."):
            answer = ask(question)
        st.write(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})