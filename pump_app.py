
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import reverse_geocoder as rg

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tanzania Water Pump Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Font & base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #065A82 0%, #1C7293 60%, #028090 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
  .hero p  { font-size: 1.05rem; color: #A8DAEC; margin-top: 0.4rem; }
  .hero .badge {
    display: inline-block;
    background: #02C39A;
    color: #021C2E;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* ── Stat cards row ── */
  .cards-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .stat-card {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(6,90,130,0.10);
    border-top: 4px solid #065A82;
  }
  .stat-card .num  { font-size: 1.9rem; font-weight: 700; color: #065A82; }
  .stat-card .lbl  { font-size: 0.78rem; color: #6B8FA8; margin-top: 2px; }

  /* ── Result box ── */
  .result-box {
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
  }
  .result-box h2 { font-size: 1.7rem; font-weight: 700; margin: 0; }
  .result-box p  { margin: 0.3rem 0 0 0; font-size: 0.95rem; }
  .functional     { background: #E6F9F3; border: 2px solid #02C39A; color: #016B52; }
  .non_functional { background: #FEE9E9; border: 2px solid #F96167; color: #8B1A1A; }
  .needs_repair   { background: #FEF9E6; border: 2px solid #F9C74F; color: #7A5C00; }

  /* ── Section header ── */
  .section-header {
    font-size: 1.05rem; font-weight: 700;
    color: #065A82; margin: 1.2rem 0 0.5rem 0;
    border-left: 4px solid #02C39A;
    padding-left: 0.6rem;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background: #F0F7FC; }
  [data-testid="stSidebar"] h2 { color: #065A82; }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

@st.cache_resource
def load_encoder():
    return joblib.load('label_encoder.pkl')

@st.cache_data
def load_zone_names():
    return sorted(joblib.load('zone_names.pkl'))

@st.cache_data
def get_zone_from_coords(lat, lon):
    result = rg.search([(lat, lon)], mode=1)[0]
    return result['admin1'] or 'Unknown'

model      = load_model()
le         = load_encoder()
zone_names = load_zone_names()


# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">EpsilonAI · CDSP Final Project</div>
  <h1>💧 Tanzania Water Pump Predictor</h1>
  <p>Predict whether a water pump is <b>Functional</b>, <b>Non-Functional</b>, or <b>Needs Repair</b> using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ── Dataset stats row ─────────────────────────────────────────────────────────
st.markdown("""
<div class="cards-row">
  <div class="stat-card"><div class="num">59,400</div><div class="lbl">Pumps in Dataset</div></div>
  <div class="stat-card"><div class="num">16</div><div class="lbl">Features Used</div></div>
  <div class="stat-card"><div class="num">3</div><div class="lbl">Status Classes</div></div>
  <div class="stat-card"><div class="num">~82%</div><div class="lbl">Model Accuracy</div></div>
  <div class="stat-card"><div class="num">XGBoost</div><div class="lbl">Algorithm</div></div>
</div>
""", unsafe_allow_html=True)

# ── Layout: inputs left, results right ───────────────────────────────────────
col_input, col_result = st.columns([1.1, 1], gap="large")

with col_input:
    st.image(
        "pump.png",
        caption="Typical Tanzanian hand pump",
        use_container_width=True,
    )

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pump Details")
    st.caption("Fill in the pump characteristics to get a prediction.")

    st.markdown('<div class="section-header">Water Characteristics</div>', unsafe_allow_html=True)
    quantity       = st.selectbox("Water Quantity",    ["enough","insufficient","dry","seasonal","unknown"])
    water_quality  = st.selectbox("Water Quality",     ["soft","salty","milky","coloured","fluoride","unknown","salty abandoned"])
    source         = st.selectbox("Water Source",      ["spring","rainwater harvesting","dam","machine dbh","shallow well","river","lake","other"])

    st.markdown('<div class="section-header">Pump Info</div>', unsafe_allow_html=True)
    waterpoint_type        = st.selectbox("Waterpoint Type",      ["communal standpipe","hand pump","other","communal standpipe multiple","improved spring","cattle trough","dam"])
    extraction_type_class  = st.selectbox("Extraction Type",      ["gravity","handpump","submersible","motorpump","rope pump","wind-powered","other"])
    payment                = st.selectbox("Payment Method",        ["never pay","per bucket","monthly","on failure","annually","unknown","other"])
    management             = st.selectbox("Management",            ["vwc","wug","water authority","wua","water board","parastatal","private operator","company","other","unknown"])

    st.markdown('<div class="section-header">Organisation</div>', unsafe_allow_html=True)
    funder    = st.selectbox("Funder",    ["government of tanzania","danida","hesawa","rwssp","world bank","kkkt","world vision","unicef","netherlands","dhv","other"])
    installer = st.selectbox("Installer", ["dwe","government","rwe","commu","danida","hesawa","kkkt","world vision","rc church","tcrs","other"])
    permit         = st.selectbox("Permit",               ["Yes","No"])
    public_meeting = st.selectbox("Public Meeting Held",  ["Yes","No"])

    st.markdown('<div class="section-header">Location & Age</div>', unsafe_allow_html=True)
    use_coords = st.toggle("Use GPS coordinates for zone", value=False)

    if use_coords:
        latitude  = st.number_input("Latitude",  min_value=-12.0, max_value=-1.0,  value=-6.0, step=0.01)
        longitude = st.number_input("Longitude", min_value=29.0,  max_value=41.0,  value=35.0, step=0.01)
        zone = get_zone_from_coords(latitude, longitude)
        st.success(f"📍 Detected zone: **{zone}**")
    else:
        zone = st.selectbox("Zone (Region)", zone_names)

    gps_height  = st.number_input("GPS Height (m)",      min_value=0,    max_value=3000,   value=500)
    population  = st.number_input("Population Nearby",   min_value=0,    max_value=30000,  value=200)
    amount_tsh  = st.number_input("Amount TSH (fee)",    min_value=0.0,  max_value=500000.0, value=0.0)
    age         = st.slider("Pump Age (years)",          min_value=0,    max_value=60,     value=10)

    predict_btn = st.button("🔍 Predict Pump Status", use_container_width=True, type="primary")


# ── Prediction ────────────────────────────────────────────────────────────────
with col_result:
    if predict_btn:
        input_df = pd.DataFrame([{
            'quantity':              quantity,
            'waterpoint_type':       waterpoint_type,
            'extraction_type_class': extraction_type_class,
            'payment':               payment,
            'water_quality':         water_quality,
            'source':                source,
            'gps_height':            gps_height,
            'population':            population,
            'management':            management,
            'funder':                funder,
            'installer':             installer,
            'permit':                int(permit == 'Yes'),
            'public_meeting':        int(public_meeting == 'Yes'),
            'amount_tsh':            amount_tsh,
            'zone':                  zone,
            'age':                   age,
        }])

        pred_enc   = model.predict(input_df)[0]
        prediction = le.inverse_transform([pred_enc])[0]
        probs      = model.predict_proba(input_df)[0]
        classes    = le.inverse_transform(model.classes_)

        # Result box
        css_class = prediction.replace(' ', '_')
        icons = {'functional': '🟢', 'non functional': '🔴', 'functional needs repair': '🟡'}
        icon  = icons.get(prediction, '⚪')

        st.markdown(f"""
        <div class="result-box {css_class}">
          <h2>{icon} {prediction.title()}</h2>
          <p>Zone detected: <b>{zone}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown('<div class="section-header">Prediction Confidence</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({'Status': classes, 'Probability': probs}) \
                    .sort_values('Probability', ascending=False)
        st.dataframe(
            prob_df.style
                .format({'Probability': '{:.1%}'})
                .bar(subset=['Probability'], color='#1C7293'),
            hide_index=True,
            use_container_width=True,
        )

        # Input summary
        st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    else:
        st.markdown("""
        <div style="background:#F0F7FC; border-radius:14px; padding:2rem; text-align:center; color:#4A7A96; margin-top:1rem;">
          <div style="font-size:3rem;">💧</div>
          <h3 style="color:#065A82;">Ready to Predict</h3>
          <p>Fill in the pump details on the left sidebar and click <b>Predict Pump Status</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            "tanzania_map.png",
            caption="Tanzania — 59,400 pumps mapped across the country",
            use_container_width=True,
        )
