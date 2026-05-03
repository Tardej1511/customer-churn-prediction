import streamlit as st
import pickle

st.set_page_config(page_title="Churn AI System", page_icon="🤖", layout="wide")

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🤖 Customer Churn AI Assistant")
st.write("Fill the form or chat with AI to predict customer churn")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
col1, col2 = st.columns(2)

# ---------------- LEFT SIDE (FORM) ----------------
with col1:
    st.subheader("📥 Customer Input Form")

    credit_score = st.number_input("Credit Score", 300, 900, 600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 200000.0, 50000.0)
    products = st.slider("Products", 1, 4, 1)
    has_card = st.selectbox("Has Card", [0, 1])
    active = st.selectbox("Active Member", [0, 1])
    salary = st.number_input("Salary", 0.0, 200000.0, 50000.0)

    if st.button("🔍 Predict from Form"):
        gender_val = 1 if gender == "Male" else 0
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0

        input_data = [[
            credit_score, gender_val, age, tenure, balance,
            products, has_card, active, salary,
            geo_germany, geo_spain
        ]]

        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            result = f"⚠️ Customer will CHURN ({round(prob*100,2)}%)"
        else:
            result = f"✅ Customer will STAY ({round(prob*100,2)}%)"

        st.success(result)

        # Add to chat
        st.session_state.messages.append({"role": "assistant", "content": result})

# ---------------- RIGHT SIDE (CHATBOT) ----------------
with col2:
    st.subheader("💬 AI Chatbot")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Enter values (comma separated)")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            values = list(map(float, user_input.split(",")))

            if len(values) != 11:
                response = "❌ Enter exactly 11 values."
            else:
                input_data = [values]
                input_data = scaler.transform(input_data)

                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]

                if pred == 1:
                    response = f"⚠️ Customer will CHURN ({round(prob*100,2)}%)"
                else:
                    response = f"✅ Customer will STAY ({round(prob*100,2)}%)"

        except:
            response = "❌ Invalid input."

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

# Footer
st.markdown("---")
st.write("Made by Your Name | AI + ML Project 🚀")
