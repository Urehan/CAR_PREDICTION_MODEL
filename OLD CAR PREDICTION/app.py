import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -------------------
# CUSTOM CSS
# -------------------
st.markdown("""
<style>

.stApp{
background:
linear-gradient(
135deg,
#0f172a,
#111827,
#1e293b
);
}

.block-container{
padding-top:1rem;
max-width:1400px;
}

.hero{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px);
border:1px solid rgba(255,255,255,0.15);
padding:30px;
border-radius:25px;
margin-bottom:20px;
text-align:center;
}

.glass{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px);
border:1px solid rgba(255,255,255,0.15);
padding:25px;
border-radius:24px;
margin-bottom:20px;
}

.price-card{
background:linear-gradient(135deg,#22c55e,#16a34a);
padding:35px;
border-radius:24px;
text-align:center;
box-shadow:0px 15px 35px rgba(0,0,0,0.3);
}

.big-price{
font-size:42px;
font-weight:800;
color:white;
}

.small-text{
color:#e5e7eb;
}

h1,h2,h3,h4,h5,h6,p,label{
color:white !important;
}

.stSelectbox label,
.stNumberInput label{
color:white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------
# MODEL
# -------------------
@st.cache_resource
def load_model():
    return joblib.load("Old_car_prediction_model.pkl")

try:
    model = load_model()
except:
    model = None

# -------------------
# HERO
# -------------------
st.markdown("""
<div class='hero'>
<h1>🚗 AI Car Price Predictor</h1>
<p>Premium Machine Learning Dashboard for Vehicle Valuation</p>
</div>
""", unsafe_allow_html=True)

# -------------------
# LAYOUT
# -------------------
left, right = st.columns([1,1])

with left:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("Vehicle Information")

    year = st.number_input(
        "Manufacturing Year",
        1990,
        datetime.now().year,
        2018
    )

    present_price = st.number_input(
        "Present Price (Lakhs)",
        0.0,
        100.0,
        5.0
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        0,
        500000,
        30000
    )

    owner = st.selectbox(
        "Number of Previous Owners",
        [0,1,2,3]
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("Vehicle Specifications")

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol","Diesel"]
    )

    seller = st.selectbox(
        "Seller Type",
        ["Dealer","Individual"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual","Automatic"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------
# PREDICTION
# -------------------
if st.button("🚀 Predict Price", use_container_width=True):

    fuel_diesel = 1 if fuel == "Diesel" else 0
    fuel_petrol = 1 if fuel == "Petrol" else 0

    seller_individual = 1 if seller == "Individual" else 0

    transmission_manual = 1 if transmission == "Manual" else 0

    features = pd.DataFrame([[
        year,
        present_price,
        kms_driven,
        owner,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]], columns=[
        'Year',
        'Present_Price',
        'Kms_Driven',
        'Owner',
        'Fuel_Type_Diesel',
        'Fuel_Type_Petrol',
        'Seller_Type_Individual',
        'Transmission_Manual'
    ])

    if model is not None:

        prediction = model.predict(features)[0]

        st.markdown(f"""
        <div class='price-card'>
        <h2>Estimated Resale Value</h2>
        <div class='big-price'>₹ {prediction:.2f} Lakhs</div>
        </div>
        """, unsafe_allow_html=True)

        confidence = min(
            100,
            max(
                45,
                100 - (kms_driven/10000)
            )
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text':'Market Confidence'},
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,40],'color':'#ef4444'},
                    {'range':[40,70],'color':'#f59e0b'},
                    {'range':[70,100],'color':'#22c55e'}
                ]
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("""
        ### 🤖 AI Vehicle Insights

        ✅ Lower mileage improves resale value

        ✅ Newer vehicles usually retain value better

        ✅ Individual seller status analyzed

        ✅ Fuel type considered in valuation

        ✅ Transmission impact evaluated

        ✅ Ownership history included
        """)

    else:
        st.error(
            "Model file not found. Place Old_car_prediction_model.pkl in project folder."
        )

# -------------------
# FOOTER
# -------------------
st.markdown("---")
st.caption("Powered by Machine Learning • Streamlit • Plotly")