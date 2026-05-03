import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="🤖", layout="wide")

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .box {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🤖 Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Decision Support Dashboard</div>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1,1])

# LEFT SIDE - INPUT
with col1:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("📥 Enter Customer Details")

    credit_score = st.number_input("Credit Score", 300, 900, 600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 200000.0, 50000.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_card = st.selectbox("Has Credit Card", [0, 1])
    active = st.selectbox("Is Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    predict_btn = st.button("🔍 Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE - OUTPUT
with col2:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict_btn:

        # Encoding
        gender_val = 1 if gender == "Male" else 0
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0

        input_data = [[
            credit_score, gender_val, age, tenure, balance,
            num_products, has_card, active, salary,
            geo_germany, geo_spain
        ]]

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        # Agent logic
        if prediction == 1:
            if balance > 100000:
                action = "Give premium offer"
            elif age > 50:
                action = "Provide loyalty benefits"
            else:
                action = "Give discount"

            st.error("⚠️ Customer is likely to CHURN")
        else:
            action = "Customer is safe"

            st.success("✅ Customer will STAY")

        st.markdown("### 💡 Suggested Action")
        st.info(action)

    else:
        st.write("👈 Enter details and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)