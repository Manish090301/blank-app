import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Set sleek page configuration
st.set_page_config(
    page_title="EduVision | Institutional Performance Matrix", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. EXPANDED DATA MODELING (RICH METRICS)
# ==========================================
@st.cache_data
def load_comprehensive_data():
    college_data = {
        "College_ID": [101, 102, 103, 104, 105, 106, 107, 108],
        "Institution_Name": ["IIT Bombay", "IIT Delhi", "IIT Kharagpur", "NIT Trichy", "NIT Surathkal", "BITS Pilani", "DTU Delhi", "IIIT Hyderabad"],
        "Tier": ["Tier 1", "Tier 1", "Tier 1", "Tier 2", "Tier 2", "Tier 1", "Tier 2", "Tier 1"],
        "NIRF_Ranking": [1, 2, 6, 9, 12, 25, 29, 15],
        "Cutoff_2023": [300, 450, 900, 4500, 5200, 1500, 7000, 2000],
        "Cutoff_2024": [320, 430, 950, 4300, 5400, 1600, 6800, 1950],
        "Cutoff_2025": [295, 420, 980, 4100, 5600, 1550, 6500, 1800],
        # Brand New Metric Columns Requested
        "Avg_Placement_LPA": [21.8, 20.5, 18.9, 15.2, 14.8, 19.2, 16.5, 23.4],
        "Highest_Package_LPA": [168.0, 150.0, 120.0, 52.0, 48.0, 60.0, 64.0, 85.0],
        "Alumni_Network_Count": [45000, 42000, 55000, 28000, 24000, 38000, 31000, 12000],
        "Notable_Alumni": ["Sundar Pichai (CEO Alphabet)", "Sachin Bansal (Flipkart)", "Arvind Kejriwal (Politician)", "Natarajan Chandrasekaran (Tata)", "K. Sivan (Ex-ISRO)", "Sabeer Bhatia (Hotmail)", "Vijay Shekhar Sharma (Paytm)", "Gagan Narang (Olympian)"],
        "Campus_Size_Acres": [550, 320, 2100, 800, 295, 328, 164, 66],
        "Research_Publications_2025": [1200, 1150, 1400, 450, 410, 680, 520, 890]
    }
    return pd.DataFrame(college_data)

df_colleges = load_comprehensive_data()

# ==========================================
# 2. MACHINE LEARNING ENGINE
# ==========================================
def predict_2026_cutoff(row):
    X = np.array([2023, 2024, 2025]).reshape(-1, 1)
    y = np.array([row['Cutoff_2023'], row['Cutoff_2024'], row['Cutoff_2025']])
    model = LinearRegression()
    model.fit(X, y)
    return max(1, int(model.predict(np.array([[2026]]))[0]))

df_colleges['Predicted_Cutoff_2026'] = df_colleges.apply(predict_2026_cutoff, axis=1)

# ==========================================
# 3. INTERACTIVE SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://img.icons8.com/fluent/96/education.png", width=80)
st.sidebar.title("EduVision Control Panel")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Dashboard View:", 
    ["🎯 Student Selection Pathway", "💼 Placement & ROI Analytics", "🌐 Alumni & Global Footprint"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip for Recruiters:** Use the tabs to toggle views instantly. All analytics update in real time.")

# ==========================================
# VIEW 1: STUDENT SELECTION PATHWAY
# ==========================================
if page == "🎯 Student Selection Pathway":
    st.title("🎯 Actionable Selection Pathway Engine")
    st.markdown("Predict institutional entry matrices and cross-reference choices using predictive machine learning paths.")
    st.write("---")
    
    # Inputs
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        student_rank = st.number_input("Enter Student's All India Rank (AIR):", min_value=1, value=1500, step=100)
    with col_in2:
        selected_tier = st.multiselect("Filter by Tier Preference:", options=["Tier 1", "Tier 2"], default=["Tier 1", "Tier 2"])
    
    df_colleges['Status'] = np.where(student_rank <= df_colleges['Predicted_Cutoff_2026'], "Highly Eligible ✅", "Reach / Ambitious ❌")
    
    # Filtration Strategy
    filtered_df = df_colleges[df_colleges['Tier'].isin(selected_tier)]
    
    # Display table
    st.subheader("📊 Dynamic Eligibility Matrix")
    st.dataframe(
        filtered_df[['Institution_Name', 'Tier', 'NIRF_Ranking', 'Predicted_Cutoff_2026', 'Status']].sort_values(by='NIRF_Ranking'), 
        width="stretch", 
        hide_index=True
    )
    
    # Visualizations
    st.write("---")
    st.subheader("⏳ Multi-Year Cutoff Trajectory Analytics")
    target_college = st.selectbox("Select Institution to Analyze Historical Trend Lines:", df_colleges['Institution_Name'].unique())
    
    crow = df_colleges[df_colleges['Institution_Name'] == target_college].iloc[0]
    trend_data = pd.DataFrame({
        "Year": [2023, 2024, 2025, 2026],
        "Cutoff Threshold": [crow['Cutoff_2023'], crow['Cutoff_2024'], crow['Cutoff_2025'], crow['Predicted_Cutoff_2026']],
        "Context": ["Historical", "Historical", "Historical", "ML Prediction"]
    })
    fig_line = px.line(trend_data, x="Year", y="Cutoff Threshold", markers=True, title=f"Admission Threshold Trajectory for {target_college}")
    fig_line.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# VIEW 2: PLACEMENT & ROI ANALYTICS
# ==========================================
elif page == "💼 Placement & ROI Analytics":
    st.title("💼 Compensation & Corporate Placement Metrics")
    st.markdown("Evaluate institutional choices by ROI metrics, package scales, and market value outcomes.")
    st.write("---")
    
    # Metric Callouts
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Highest CTC Recorded Across Pool", value=f"₹ {df_colleges['Highest_Package_LPA'].max()} LPA")
    with m2:
        st.metric(label="Highest Average Package", value=f"₹ {df_colleges['Avg_Placement_LPA'].max()} LPA", delta="IIIT Hyderabad")
    with m3:
        st.metric(label="Average Cohort Compensation", value=f"₹ {round(df_colleges['Avg_Placement_LPA'].mean(), 1)} LPA")
        
    st.write("---")
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 Average vs. Highest Package Distribution (LPA)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=df_colleges['Institution_Name'], y=df_colleges['Avg_Placement_LPA'], name='Average Package (LPA)', marker_color='#1f77b4'))
        fig_bar.add_trace(go.Bar(x=df_colleges['Institution_Name'], y=df_colleges['Highest_Package_LPA'], name='Highest Package (LPA)', marker_color='#ff7f0e'))
        fig_bar.update_layout(barmode='group', xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("🎯 Infrastructure Scale vs. Research Volume")
        fig_bubble = px.scatter(
            df_colleges, x="Campus_Size_Acres", y="Research_Publications_2025",
            size="Avg_Placement_LPA", color="Tier", hover_name="Institution_Name",
            labels={"Campus_Size_Acres": "Campus Area (Acres)", "Research_Publications_2025": "Annual Research Publications"},
            title="Size corresponds to Average LPA Package Value"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

# ==========================================
# VIEW 3: ALUMNI & GLOBAL FOOTPRINT
# ==========================================
elif page == "🌐 Alumni & Global Footprint":
    st.title("🌐 Alumni Networks & Academic Footprint Strength")
    st.markdown("Track the institutional scale beyond graduation through alumni size, flagship leaders, and scientific output.")
    st.write("---")
    
    # Highlight Cards
    st.subheader("🏆 Key Network Leaders & Influencers")
    card_cols = st.columns(4)
    for idx, row in df_colleges.head(4).iterrows():
        with card_cols[idx % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{row['Institution_Name']}</h4>
                <p><b>Notable Star Alumni:</b><br>{row['Notable_Alumni']}</p>
                <p><b>Active Network Size:</b> {row['Alumni_Network_Count']:,} grads</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.write("---")
    
    # Network Size Breakdown
    st.subheader("📊 Active Alumni Matrix Capacities")
    fig_pie = px.pie(
        df_colleges, values='Alumni_Network_Count', names='Institution_Name',
        title='Global Registered Alumni Share Grouping',
        hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Raw Data Export Feature
    st.write("---")
    st.subheader("📥 Export Institutional Metric Ledger")
    st.markdown("Download this comprehensive consolidated multi-criteria institutional performance map for reference or reporting pipelines.")
    
    csv_buffer = df_colleges.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Structured CSV Report Ledger",
        data=csv_buffer,
        file_name="Institutional_Performance_Ledger_2026.csv",
        mime="text/csv"
    )
