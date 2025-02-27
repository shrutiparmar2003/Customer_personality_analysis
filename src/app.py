import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model_path = "models/random_forest_best.pkl"
best_rf = joblib.load(model_path)

# Apply Custom CSS for Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        * {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: linear-gradient(to right, #E3FDFD, #FFFFFF);
        }

        .css-1cpxqw2 {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }

        .stButton>button {
            background: linear-gradient(90deg, #007BFF, #00C9FF);
            border: none;
            color: white;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
        }
        
        .stButton>button:hover {
            background: linear-gradient(90deg, #00C9FF, #007BFF);
            transform: scale(1.05);
        }

        h1, h2, h3, h4 {
            color: #005A9C !important;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Detect if user is in dark mode
is_dark_mode = st.get_option("theme.base") == "dark"

# Define text colors based on mode
success_color = "#28a745" if not is_dark_mode else "#90EE90"  # Green
warning_color = "#FFC107" if not is_dark_mode else "#FFD700"  # Yellow

# Streamlit UI with New Elegant Design
st.image("https://img.freepik.com/free-vector/customers-earning-money-by-giving-likes_74855-7121.jpg?t=st=1740616499~exp=1740620099~hmac=5673f11f56987b696ada5e1ee65d7ab88c224b7ad86dbd3029e3d7c34cf94633&w=2000", use_container_width=True)

st.title("📊 Customer Response Prediction")
st.markdown("**Predict whether a customer will respond to a marketing campaign.**")

# Cluster descriptions
cluster_info = {
    0: "🌟 High spenders, frequent buyers",
    1: "🛍️ Moderate spenders, occasional buyers",
    2: "🛒 Low spenders, but engaged customers",
    3: "📉 Least engaged, low spending, high recency"
}

# **📌 Single Customer Prediction**
st.subheader("🔹 Single Customer Prediction")

# User Input Form
total_spending = st.number_input("💰 Total Spending ($)", min_value=0, max_value=10000, value=500, step=100)
recency = st.number_input("📅 Recency (days since last purchase)", min_value=0, max_value=365, value=30, step=5)
cluster = st.selectbox("🎯 Customer Cluster", list(cluster_info.keys()), format_func=lambda x: cluster_info[x])
age = st.number_input("👤 Age", min_value=18, max_value=100, value=35, step=1)

# Prediction Button
if st.button("🔍 Predict Response"):
    # Prepare input data
    input_data = pd.DataFrame([[total_spending, recency, cluster, age]], 
                              columns=["Total_Spending", "Recency", "Cluster", "Age"])
    
    # Get probability
    probability = best_rf.predict_proba(input_data)[:, 1][0]

    # Adjust threshold (0.4)
    predicted_response = 1 if probability >= 0.4 else 0

    # Display Result
    st.subheader("📊 Prediction Result")
    if predicted_response == 1:
        st.markdown(
            f'<p style="color: {success_color}; font-size:18px; font-weight: bold;">'
            f"✅ This customer is <b>likely to respond</b> (Probability: {probability:.2f})</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<p style="color: {warning_color}; font-size:18px; font-weight: bold;">'
            f"❌ This customer is <b>unlikely to respond</b> (Probability: {probability:.2f})</p>",
            unsafe_allow_html=True
        )

# **📌 Bulk Upload Prediction**
st.subheader("📂 Bulk Upload for Multiple Customers")

uploaded_file = st.file_uploader("📥 Upload a CSV file with customer data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove extra spaces from column names

    # Check if required columns exist
    required_columns = ["Total_Spending", "Recency", "Cluster", "Age"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ CSV file must contain these columns: {', '.join(required_columns)}")
    else:
        # Get probabilities
        probabilities = best_rf.predict_proba(df)[:, 1]

        # Apply threshold
        df["Predicted_Response"] = (probabilities >= 0.4).astype(int)
        df["Response_Probability"] = probabilities

        # Show results
        st.write("✅ Prediction Results:", df)

        # Save as downloadable CSV
        df.to_csv("data/predicted_customers.csv", index=False)
        st.download_button(label="⬇️ Download Predictions", data=df.to_csv(index=False), 
                           file_name="predicted_customers.csv", mime="text/csv")

        # **📌 Cost Savings Analysis**
        st.subheader("💰 Cost Savings Analysis")

        total_customers = len(df)
        predicted_respondents = df["Predicted_Response"].sum()
        original_cost = total_customers * 10  # Assume $10 per customer campaign
        new_cost = predicted_respondents * 10
        savings = original_cost - new_cost

        st.write(f"📊 **Total Customers:** {total_customers}")
        st.write(f"✅ **Predicted Respondents:** {predicted_respondents}")
        st.write(f"💰 **Original Marketing Cost:** ${original_cost}")
        st.write(f"🔻 **New Cost (Targeting Only Respondents):** ${new_cost}")
        st.success(f"💸 **Estimated Savings: ${savings}**")
