import sys
from pathlib import Path

# Add project root to python path to resolve api and src modules
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set up page configurations
st.set_page_config(
    page_title="F1 Pit Stop Strategy Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pirelli official compound hex colors
PIRELLI_COLORS = {
    "SOFT": "#e10600",         # Pirelli Red
    "MEDIUM": "#ffd100",       # Pirelli Yellow
    "HARD": "#f0f0f0",         # Pirelli White/Light Grey
    "INTERMEDIATE": "#00a650",  # Pirelli Green
    "WET": "#005aff"           # Pirelli Blue
}

# Custom F1 Dark Red Styling
st.markdown("""
<style>
    .main {
        background-color: #0f1115;
        color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        font-weight: 700;
        font-size: 16px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #a0a6b5;
    }
    .stTabs [data-baseweb="tab"]:focus {
        color: #e10600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #e10600;
        border-bottom-color: #e10600;
        border-bottom-width: 3px;
    }
    .metric-card {
        background-color: #1a1c23;
        border-left: 4px solid #e10600;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-label {
        font-size: 13px;
        color: #a0a6b5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-green {
        background: linear-gradient(135deg, #1b3a24 0%, #0d1e13 100%);
        border: 1px solid #00a650;
        border-radius: 6px;
        padding: 20px;
        color: #e6f7eb;
    }
    .status-orange {
        background: linear-gradient(135deg, #3d2a14 0%, #22170b 100%);
        border: 1px solid #ffd100;
        border-radius: 6px;
        padding: 20px;
        color: #fffaf0;
    }
    .status-red {
        background: linear-gradient(135deg, #441010 0%, #250909 100%);
        border: 1px solid #e10600;
        border-radius: 6px;
        padding: 20px;
        color: #ffebeb;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(225, 6, 0, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(225, 6, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(225, 6, 0, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Lazy import so we don't fail if model isn't built yet
@st.cache_resource
def get_predictor():
    try:
        from api.predictor import F1Predictor
        return F1Predictor()
    except Exception as e:
        st.sidebar.error(f"Error loading predictor assets: {e}")
        return None

predictor = get_predictor()

# App Header
st.markdown("<h1 style='color: #e10600; margin-bottom: 0px;'>🏎️ F1 PIT STOP STRATEGY CENTER</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #a0a6b5; margin-top: 5px; font-size: 15px;'>Predicting optimal pit stop windows using a GBDT Ensemble (LightGBM + CatBoost + XGBoost)</p>", unsafe_allow_html=True)
st.markdown("---")

# Main Page Tabs
tab1, tab2 = st.tabs(["📊 BATCH STRATEGY ANALYSIS (CSV UPLOAD)", "🎯 SINGLE LAP TELEMETRY CALCULATOR"])

# ==========================================
# TAB 1: BATCH CSV UPLOADER
# ==========================================
with tab1:
    st.markdown("### Upload Telemetry File")
    st.write("Upload a race telemetry CSV file containing laps to generate pit stop predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file...", type="csv", label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file: {uploaded_file.name} ({len(raw_df):,} rows)")
            
            if predictor is None:
                st.error("Prediction engine assets are currently offline. Please run the model training script `src/pipeline.py` first.")
            else:
                with st.spinner("Processing race telemetry and generating ensemble predictions..."):
                    # Process prediction
                    results_df = predictor.predict_csv(raw_df)
                    
                    # Merge results with input for visualization
                    df_combined = raw_df.copy()
                    df_combined['pit_probability'] = results_df['pit_probability']
                    df_combined['should_pit'] = results_df['should_pit']
                    
                    # Format submission compliant output
                    if 'id' in df_combined.columns:
                        submission_df = pd.DataFrame({
                            'id': df_combined['id'],
                            'PitNextLap': df_combined['pit_probability']
                        }).sort_values('id').reset_index(drop=True)
                    else:
                        submission_df = pd.DataFrame({
                            'id': np.arange(len(df_combined)),
                            'PitNextLap': df_combined['pit_probability']
                        })

                # Display Dashboard Cards
                st.markdown("#### Strategy Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                total_laps = len(df_combined)
                pit_laps = df_combined['should_pit'].sum()
                avg_prob = df_combined['pit_probability'].mean() * 100
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Race Laps Analyzed</div>
                        <div class="metric-value">{total_laps:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{pit_laps:,} ({pit_laps/total_laps:.1%})</div>
                        <div class="metric-label">Predicted Pit Stops</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_prob:.2f}%</div>
                        <div class="metric-label">Average Pit Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    # Top drivers likely to pit
                    df_sorted = df_combined.sort_values(by="pit_probability", ascending=False)
                    top_driver = df_sorted.iloc[0]['Driver'] if len(df_sorted) > 0 else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{top_driver}</div>
                        <div class="metric-label">Critical Strat. Alert</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Layout for charts and data
                col_chart, col_data = st.columns([2, 1])

                with col_chart:
                    st.markdown("#### Interactive Strategy Map: Tyre Age vs Pit Probability")
                    
                    # Pirelli color mapper
                    fig = px.scatter(
                        df_combined,
                        x="TyreLife",
                        y="pit_probability",
                        color="Compound",
                        color_discrete_map=PIRELLI_COLORS,
                        hover_data=["Driver", "Race", "LapNumber", "Stint"],
                        labels={
                            "TyreLife": "Tyre Age (Laps)",
                            "pit_probability": "Blended Pit Stop Probability",
                            "Compound": "Tyre Compound"
                        },
                        template="plotly_dark"
                    )
                    
                    fig.update_layout(
                        plot_bgcolor="#16181c",
                        paper_bgcolor="#0f1115",
                        xaxis=dict(showgrid=True, gridcolor="#2c2e35"),
                        yaxis=dict(showgrid=True, gridcolor="#2c2e35", range=[0, 1.05]),
                        legend=dict(bgcolor="rgba(0,0,0,0)")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_data:
                    st.markdown("#### Download Predictions")
                    st.write("Download the pit stop strategy predictions featuring driver IDs and blended probabilities.")
                    
                    csv_data = submission_df.to_csv(index=False)
                    st.download_button(
                        label="💾 DOWNLOAD PIT STRATEGY PREDICTIONS (CSV)",
                        data=csv_data,
                        file_name="pit_strategy_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### Top 5 Critical Strategic Lap Alerts")
                    alerts_df = df_combined.sort_values(by="pit_probability", ascending=False).head(5)
                    
                    for _, row in alerts_df.iterrows():
                        st.markdown(f"""
                        <div style="background-color: #1a1c23; border-left: 3px solid #e10600; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                            <span style="font-weight:700; color: #fff;">{row['Driver']}</span> @ {row['Race']} <br>
                            Lap {int(row['LapNumber'])} | Compound: <span style="color:{PIRELLI_COLORS.get(row['Compound'], '#fff')}">{row['Compound']}</span> | Tyre Age: {int(row['TyreLife'])} Laps<br>
                            <span style="color:#e10600; font-weight:700;">Pit Prob: {row['pit_probability']:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

# ==========================================
# TAB 2: SINGLE MANUAL TELEMETRY FORM
# ==========================================
with tab2:
    st.markdown("### Telemetry Strategy Simulator")
    st.write("Manually enter telemetry values for a single lap to instantly calculate if a driver is inside their optimal pit stop window.")
    
    # Check if models are available
    if predictor is None:
        st.warning("Prediction engine assets are offline. Please run the model training script `src/pipeline.py` first.")
    else:
        # Load lists of Drivers and Races if available from encoder mappings, otherwise use defaults
        known_drivers = sorted(list(predictor.target_encoders.get('Driver', {}).get('map', {}).keys()))
        known_races = sorted(list(predictor.target_encoders.get('Race', {}).get('map', {}).keys()))
        
        if not known_drivers:
            known_drivers = ["Verstappen", "Hamilton", "Leclerc", "Norris", "Sainz", "Perez", "Russell", "Alonso", "Piastri", "Gasly"]
        if not known_races:
            known_races = ["Monaco", "Spa", "Silverstone", "Monza", "Jeddah", "Singapore", "Suzuka", "Austin", "Melbourne", "Abu Dhabi"]

        # Form layout
        col_form, col_indicator = st.columns([1, 1])
        
        with col_form:
            st.markdown("#### Telemetry Inputs")
            
            form_driver = st.selectbox("Driver", known_drivers)
            form_race = st.selectbox("Race Venue", known_races)
            form_compound = st.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
            
            c_left, c_right = st.columns(2)
            with c_left:
                form_lap = st.number_input("Current Lap Number", min_value=1, max_value=85, value=25)
                form_stint = st.number_input("Stint Number", min_value=1, max_value=6, value=1)
                form_position = st.number_input("Race Position", min_value=1, max_value=20, value=5)
                form_pos_change = st.number_input("Position Change vs Last Lap", min_value=-10, max_value=10, value=0)
            
            with c_right:
                form_tyrelife = st.slider("Tyre Age (Laps Run)", min_value=0.0, max_value=45.0, value=15.0, step=1.0)
                form_laptime = st.number_input("Lap Time (s)", min_value=50.0, max_value=180.0, value=82.5, step=0.1)
                form_laptime_delta = st.number_input("Lap Time Delta vs Last Lap (s)", min_value=-5.0, max_value=10.0, value=0.1, step=0.01)
                form_degrad = st.number_input("Cumulative Tyre Degradation", min_value=0.0, max_value=2.0, value=0.12, step=0.01)

            # Auto calculate RaceProgress based on approximate venue laps
            typical_laps = {
                "Monaco": 78, "Spa": 44, "Silverstone": 52, "Monza": 53, 
                "Melbourne": 58, "Abu Dhabi": 58, "Singapore": 62, "Suzuka": 53
            }
            max_laps = typical_laps.get(form_race, 55)
            progress = min(form_lap / max_laps, 1.0)
            
            st.info(f"Auto-calculated Race Progress: {progress:.1%} (Estimated {max_laps} total laps for {form_race})")
            
            # Predict Trigger
            submit_prediction = st.button("🔥 ANALYZE STRATEGY & PREDICT", use_container_width=True)

        with col_indicator:
            st.markdown("#### Blended Strategy Recommendation")
            
            # Default values before clicking predict
            if not submit_prediction:
                st.write("Fill in the telemetry values and click the button to run the GBDT ensemble predictor.")
                
                # Render a template gray gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = 0,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Pit Stop Probability (%)", 'font': {'size': 18, 'color': '#a0a6b5'}},
                    number = {'font': {'size': 48, 'color': '#a0a6b5'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a0a6b5"},
                        'bar': {'color': "#2c2e35"},
                        'bgcolor': "#16181c",
                        'bordercolor': "#2c2e35"
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                with st.spinner("Analyzing parameters..."):
                    # Assemble single telemetry record
                    single_telemetry = {
                        "Driver": form_driver,
                        "Race": form_race,
                        "Compound": form_compound,
                        "LapNumber": int(form_lap),
                        "Stint": int(form_stint),
                        "TyreLife": float(form_tyrelife),
                        "Position": float(form_position),
                        "LapTime (s)": float(form_laptime),
                        "LapTime_Delta": float(form_laptime_delta),
                        "Cumulative_Degradation": float(form_degrad),
                        "RaceProgress": float(progress),
                        "Position_Change": float(form_pos_change),
                        "PitStop": 0.0,
                        "Year": 2025
                    }

                    # Predict
                    res = predictor.predict_single(single_telemetry)
                    prob = res['pit_probability']
                    should_pit = res['should_pit']
                    threshold = res['optimal_threshold']
                    
                    # Dynamic compound coloring
                    compound_color = PIRELLI_COLORS.get(form_compound, "#ffffff")

                    # Draw gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Pit Stop Probability (%)", 'font': {'size': 18, 'color': '#ffffff'}},
                        number = {'font': {'size': 54, 'color': '#ffffff'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a0a6b5"},
                            'bar': {'color': compound_color},
                            'bgcolor': "#16181c",
                            'bordercolor': "#2c2e35",
                            'threshold': {
                                'line': {'color': "red", 'width': 3},
                                'thickness': 0.75,
                                'value': threshold * 100
                            }
                        }
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)

                    # Dynamic recommendation card
                    st.markdown("---")
                    if prob < 0.35:
                        st.markdown(f"""
                        <div class="status-green">
                            <h4 style="margin-top:0px; color:#00a650;">🟢 TYRES HEALTHY: LOW RISK</h4>
                            <p style="margin-bottom:0px; font-size:14px;">
                                Predicted pit probability is <b>{prob:.1%}</b> (Decision Threshold is {threshold:.1%}). <br>
                                The driver is safely outside their strategic pit window. We recommend <b>staying out</b> on the current <span style="color:{compound_color}; font-weight:700;">{form_compound}</span> tyres.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prob < threshold:
                        st.markdown(f"""
                        <div class="status-orange">
                            <h4 style="margin-top:0px; color:#ffd100;">🟡 STRATEGIC PREPARATION</h4>
                            <p style="margin-bottom:0px; font-size:14px;">
                                Predicted pit probability is <b>{prob:.1%}</b>. The driver is approaching the decision threshold of <b>{threshold:.1%}</b>. <br>
                                While not yet at critical degradation, tyre performance is beginning to decline. Strategists should monitor gaps and prepare for a pit stop within the next 2-3 laps.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-red">
                            <h4 style="margin-top:0px; color:#e10600; font-weight:800;">🚨 CRITICAL STRATEGY ALERT: PIT NEXT LAP!</h4>
                            <p style="margin-bottom:0px; font-size:14px;">
                                Predicted pit probability is <b>{prob:.1%}</b>, exceeding the safety threshold of <b>{threshold:.1%}</b>. <br>
                                Tyre age ({int(form_tyrelife)} laps) and degradation are critical. <b>Pitting on the next lap is highly recommended</b> to prevent massive time loss or tyres dropping off the degradation cliff!
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Stint metrics detail
                    historical_avg = predictor.avg_stint_map.get((form_driver, form_race), None)
                    if historical_avg:
                        pct_used = form_tyrelife / historical_avg
                        st.markdown(f"""
                        <div style="background-color: #1a1c23; border-radius: 4px; padding: 12px; margin-top: 15px; font-size:13px; color: #a0a6b5;">
                            🏁 <b>Historical Benchmark</b>: {form_driver} running at {form_race} typically averages a stint of <b>{historical_avg:.1f} laps</b>.<br>
                            Current tyre age is at <b>{pct_used:.1%}</b> of their typical stint length.
                        </div>
                        """, unsafe_allow_html=True)
