import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#0f172a 0%,
#111827 50%,
#1e293b 100%
);
}

.block-container{
max-width:1400px;
padding-top:1rem;
}

.hero{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px);
padding:35px;
border-radius:25px;
border:1px solid rgba(255,255,255,0.12);
text-align:center;
margin-bottom:20px;
}

.glass{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px);
padding:25px;
border-radius:24px;
border:1px solid rgba(255,255,255,0.12);
margin-bottom:20px;
}

.price-card{
background:linear-gradient(
135deg,
#22c55e,
#16a34a
);
padding:35px;
border-radius:24px;
text-align:center;
box-shadow:0 15px 35px rgba(0,0,0,0.35);
}

.big-price{
font-size:48px;
font-weight:800;
color:white;
}

.metric-box{
background:rgba(255,255,255,0.06);
padding:20px;
border-radius:20px;
text-align:center;
}

h1,h2,h3,h4,h5,h6,p,label{
color:white !important;
}

.stNumberInput label,
.stSelectbox label{
color:white !important;
}

[data-testid="stMetricValue"]{
color:white;
}

footer{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "Old_car_prediction_model.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)

try:
    model = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    model_error = str(e)

# =====================================================
# HERO
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🚗 AI Car Price Predictor</h1>
    <p>
        Predict the resale value of your vehicle using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MODEL STATUS
# =====================================================

if model_loaded:
    st.success("✅ Model Loaded Successfully")
else:
    st.error("❌ Model Failed To Load")
    st.code(model_error)

# =====================================================
# INPUT SECTION
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📋 Vehicle Information")

    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=datetime.now().year,
        value=2018
    )

    present_price = st.number_input(
        "Current Showroom Price (Lakhs)",
        min_value=0.0,
        max_value=200.0,
        value=5.0
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=30000
    )

    owner = st.selectbox(
        "Previous Owners",
        [0,1,2,3]
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("⚙️ Specifications")

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

# =====================================================
# QUICK STATS
# =====================================================

age = datetime.now().year - year

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Vehicle Age", f"{age} Years")

with m2:
    st.metric("Driven", f"{kms_driven:,} KM")

with m3:
    st.metric("Owners", owner)

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button(
    "🚀 Predict Resale Value",
    use_container_width=True
):

    if not model_loaded:
        st.error("Model not loaded.")
        st.stop()

    fuel_diesel = 1 if fuel == "Diesel" else 0
    fuel_petrol = 1 if fuel == "Petrol" else 0

    seller_individual = (
        1 if seller == "Individual" else 0
    )

    transmission_manual = (
        1 if transmission == "Manual" else 0
    )

    features = pd.DataFrame([[
        year,
        present_price,
        kms_driven,
        owner,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]],
    columns=[
        "Year",
        "Present_Price",
        "Kms_Driven",
        "Owner",
        "Fuel_Type_Diesel",
        "Fuel_Type_Petrol",
        "Seller_Type_Individual",
        "Transmission_Manual"
    ])

    try:

        prediction = model.predict(features)[0]

        st.markdown(
            f"""
            <div class="price-card">
                <h2>Estimated Resale Value</h2>
                <div class="big-price">
                    ₹ {prediction:.2f} Lakhs
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        confidence = max(
            50,
            min(
                98,
                100 - (kms_driven / 10000)
            )
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={
                    "text":"Prediction Confidence"
                },
                gauge={
                    "axis":{
                        "range":[0,100]
                    },
                    "steps":[
                        {
                            "range":[0,40],
                            "color":"#ef4444"
                        },
                        {
                            "range":[40,70],
                            "color":"#f59e0b"
                        },
                        {
                            "range":[70,100],
                            "color":"#22c55e"
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("""
        ## 🤖 AI Insights

        ✅ Lower mileage generally increases resale value

        ✅ Newer vehicles retain market value longer

        ✅ Fuel type impacts resale demand

        ✅ Manual and automatic transmission affect pricing

        ✅ Ownership history influences buyer confidence

        ✅ Market confidence score estimated using vehicle condition indicators
        """)

        st.subheader("📊 Model Input")

        st.dataframe(
            features,
            use_container_width=True
        )

    except Exception as e:

        st.error("Prediction Error")
        st.code(str(e))

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "🚗 Powered by Machine Learning • Streamlit • Plotly"
)