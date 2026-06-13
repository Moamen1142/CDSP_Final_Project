
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load artifacts ────────────────────────────────────────────────────────────
model     = joblib.load('best_model.pkl')
kmeans    = joblib.load('kmeans_zones.pkl')
zone_names = joblib.load('zone_names.pkl')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='Tanzania Water Pump Predictor', page_icon='💧', layout='centered')
st.title('💧 Tanzania Water Pump Status Predictor')
st.markdown('Fill in the pump details below to predict whether it is **functional**, **non-functional**, or **needs repair**.')

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header('Pump Details')

quantity = st.sidebar.selectbox('Water Quantity', ['enough', 'insufficient', 'dry', 'seasonal', 'unknown'])

waterpoint_type = st.sidebar.selectbox('Waterpoint Type', [
    'communal standpipe', 'hand pump', 'other', 'communal standpipe multiple',
    'improved spring', 'cattle trough', 'dam'
])

extraction_type_class = st.sidebar.selectbox('Extraction Type', [
    'gravity', 'handpump', 'submersible', 'motorpump', 'rope pump',
    'wind-powered', 'other'
])

payment = st.sidebar.selectbox('Payment Method', [
    'never pay', 'per bucket', 'monthly', 'on failure', 'annually',
    'unknown', 'other'
])

water_quality = st.sidebar.selectbox('Water Quality', [
    'soft', 'salty', 'milky', 'coloured', 'fluoride', 'unknown', 'salty abandoned'
])

source = st.sidebar.selectbox('Water Source', [
    'spring', 'rainwater harvesting', 'dam', 'machine dbh', 'shallow well',
    'river', 'lake', 'other'
])

management = st.sidebar.selectbox('Management', [
    'vwc', 'wug', 'water authority', 'wua', 'water board', 'parastatal',
    'private operator', 'company', 'other', 'unknown'
])

funder = st.sidebar.selectbox('Funder', [
    'government of tanzania', 'danida', 'hesawa', 'rwssp', 'world bank',
    'kkkt', 'world vision', 'unicef', 'netherlands', 'dhv', 'other'
])

installer = st.sidebar.selectbox('Installer', [
    'dwe', 'government', 'rwe', 'commu', 'danida', 'hesawa', 'kkkt',
    'world vision', 'rc church', 'tcrs', 'other'
])

permit = st.sidebar.selectbox('Permit', ['Yes', 'No'])
public_meeting = st.sidebar.selectbox('Public Meeting Held', ['Yes', 'No'])

gps_height = st.sidebar.number_input('GPS Height (m)', min_value=0, max_value=3000, value=500)
population  = st.sidebar.number_input('Population Nearby', min_value=0, max_value=30000, value=200)
amount_tsh  = st.sidebar.number_input('Amount TSH (fee)', min_value=0.0, max_value=500000.0, value=0.0)
age         = st.sidebar.slider('Pump Age (years)', min_value=0, max_value=60, value=10)

longitude   = st.sidebar.number_input('Longitude', min_value=29.0, max_value=41.0, value=35.0)
latitude    = st.sidebar.number_input('Latitude',  min_value=-12.0, max_value=-1.0, value=-6.0)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.sidebar.button('🔍 Predict'):

    # compute zone from coordinates
    zone_id   = kmeans.predict([[longitude, latitude]])[0]
    zone_label = zone_names.get(zone_id, 'Other')

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
        'zone':                  zone_label,
        'age':                   age,
    }])

    prediction   = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes      = model.classes_

    # ── Result display ────────────────────────────────────────────────────────
    color = {'functional': '🟢', 'non functional': '🔴', 'functional needs repair': '🟡'}
    st.markdown(f'## {color.get(prediction, "⚪")} Prediction: **{prediction.title()}**')
    st.markdown(f'**Assigned Zone:** {zone_label}')

    st.subheader('Prediction Probabilities')
    prob_df = pd.DataFrame({'Status': classes, 'Probability': probabilities}).sort_values('Probability', ascending=False)
    st.bar_chart(prob_df.set_index('Status'))

    st.subheader('Input Summary')
    st.dataframe(input_df)

else:
    st.info('👈 Fill in the pump details on the left and click **Predict**.')
