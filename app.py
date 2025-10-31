import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import altair as alt

# --------------------------------------------------------
# 🌆 PAGE SETUP
# --------------------------------------------------------
st.set_page_config(
    page_title="Jakarta AQI Forecast Dashboard",
    page_icon="🌇",
    layout="wide"
)

# --------------------------------------------------------
# ⚙️ LOAD MODEL AND DATA (with caching)
# --------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("jakarta_aqi_model.pkl")

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("cleaned_jakarta_aqi.csv")

model = load_model()
df = load_data()

# --------------------------------------------------------
# 📍 DEFINE STATION COORDINATES
# --------------------------------------------------------
station_coords = {
    "DKI1 (Bunderan HI)": (-6.193, 106.820),
    "DKI2 (Kelapa Gading)": (-6.166, 106.909),
    "DKI3 (Jagakarsa)": (-6.338, 106.823),
    "DKI4 (Lubang Buaya)": (-6.293, 106.894),
    "DKI5 (Kebon Jeruk)": (-6.200, 106.770),
}

# --------------------------------------------------------
# 🧭 USER INTERFACE SETUP
# --------------------------------------------------------
st.title("🌆 Jakarta Air Quality Forecast Dashboard (Multi-Station)")
st.markdown("""
Forecasts future AQI (2025–2030) across Jakarta’s five main monitoring stations,
based on environmental and policy improvement scenarios.
""")

st.sidebar.header("🔧 Adjust Scenario Parameters")
ev_switch = st.sidebar.slider("EV Adoption (%)", 0, 100, 30)
emission_reg = st.sidebar.slider("Emission Regulation Strictness (%)", 0, 100, 40)
green_area = st.sidebar.slider("Green Area Increase (%)", 0, 100, 25)
carbon_capture = st.sidebar.slider("Carbon Capture Efficiency (%)", 0, 100, 20)

# --------------------------------------------------------
# 🤖 PREDICTION LOGIC
# --------------------------------------------------------
if st.button("🚀 Predict Future AQI"):
    future_years = np.arange(2025, 2031)
    results = []

    for station in df["stasiun"].unique():
        sub = df[df["stasiun"] == station]
        if sub.empty:
            st.warning(f"No data for {station}. Skipping.")
            continue

        mean_features = sub[["pm25", "pm10", "so2", "co", "o3", "no2"]].mean()

        # Create base feature set for each year
        X_future = pd.DataFrame([mean_features.values] * len(future_years), columns=mean_features.index)

        # 🧠 Make model react to scenario parameters
        X_future["pm25"] *= (1 - (ev_switch + emission_reg) / 200)  # EV + regulation reduce PM
        X_future["pm10"] *= (1 - (green_area + emission_reg) / 220)  # green area + reg
        X_future["co"] *= (1 - (carbon_capture + ev_switch) / 250)  # EV + capture reduce CO
        X_future["no2"] *= (1 - (emission_reg + ev_switch) / 180)  # stricter regulation = lower NO2
        X_future["so2"] *= (1 - (carbon_capture + emission_reg) / 230)
        X_future["o3"] *= (1 - (green_area) / 200)

        # Base model prediction
        base_pred = model.predict(X_future)

        # Add gradual improvement trend toward 2030
        adjusted_pred = base_pred * np.linspace(1, 0.85, len(base_pred))

        lat, lon = station_coords.get(station, (-6.2, 106.8))
        results.append({
            "stasiun": station,
            "latitude": lat,
            "longitude": lon,
            "avg_aqi": np.mean(adjusted_pred),
            "predictions": adjusted_pred
        })

    st.session_state["station_results"] = pd.DataFrame(results)
    st.session_state["selected_year"] = 2025  # default start

# --------------------------------------------------------
# 📊 DISPLAY FORECAST RESULTS
# --------------------------------------------------------
if "station_results" in st.session_state:
    results_df = st.session_state["station_results"]

    # Table
    st.subheader("📊 Average AQI per Station (2025–2030)")
    st.dataframe(results_df[["stasiun", "avg_aqi"]].style.format({"avg_aqi": "{:.1f}"}))

    # Line Chart
    st.subheader("📈 Predicted AQI Trend (2025–2030)")
    line_data = pd.DataFrame(
        {row["stasiun"]: row["predictions"] for _, row in results_df.iterrows()},
        index=np.arange(2025, 2031)
    )
    df_long = line_data.reset_index().melt(id_vars="index", var_name="Station", value_name="AQI")
    df_long.rename(columns={"index": "Year"}, inplace=True)
    chart = alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X("Year:O", axis=alt.Axis(title="Year", labelAngle=0)),
        y=alt.Y("AQI:Q", axis=alt.Axis(title="Predicted AQI")),
        color=alt.Color("Station:N", legend=alt.Legend(title="Station")),
        tooltip=["Year", "Station", "AQI"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # --------------------------------------------------------
    # 🗺️ FORECASTED AQI MAP (with inline year buttons)
    # --------------------------------------------------------
    st.subheader("🗺️ Forecasted AQI Map of Jakarta (All 5 Stations)")
    st.markdown("### Select forecast year to view on map:")

    # Inline button styling
    st.markdown("""
        <style>
        div[data-testid="column"] {
            width: auto !important;
            flex: 0 0 auto !important;
            padding-right: 0.3rem !important;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.3rem !important;
        }
        button[kind="secondary"] {
            background-color: #262730 !important;
            color: #fafafa !important;
            border: 1px solid #444 !important;
            border-radius: 6px !important;
            padding: 4px 12px !important;
        }
        button[kind="secondary"]:hover {
            background-color: #1f77b4 !important;
            border-color: #1f77b4 !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    years = list(range(2025, 2031))
    if "selected_year" not in st.session_state:
        st.session_state["selected_year"] = 2025

    cols = st.columns(len(years))
    for i, year in enumerate(years):
        with cols[i]:
            if st.button(f"{year}", key=f"year_{year}"):
                st.session_state["selected_year"] = year

    selected_year = st.session_state["selected_year"]
    year_index = selected_year - 2025
    st.markdown(f"#### 🗓 Showing forecast for **{selected_year}**")

    # Map generation
    m = folium.Map(location=[-6.2, 106.8], zoom_start=10)
    for _, row in results_df.iterrows():
        year_aqi = row["predictions"][year_index]
        color = "green" if year_aqi < 50 else "orange" if year_aqi < 100 else "red"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"<b>{row['stasiun']}</b><br>Forecasted AQI ({selected_year}): {year_aqi:.1f}",
            tooltip=f"{row['stasiun']} ({selected_year})",
            icon=folium.Icon(color=color, icon="cloud")
        ).add_to(m)
    st_folium(m, width=700, height=500)

else:
    st.info("👆 Adjust the sliders and click **Predict Future AQI** to view multi-station results.")
