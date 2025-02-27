import joblib
import pandas as pd

# Load the saved model
model_path = "models/random_forest_best.pkl"
best_rf = joblib.load(model_path)

print("✅ Model loaded successfully!")

# Sample customer data for prediction
sample_data = pd.DataFrame({
    "Total_Spending": [500, 1200, 2000, 3000],  # Varying spending levels
    "Recency": [10, 45, 5, 60],  # Recent vs. old customers
    "Cluster": [1, 3, 2, 0],  # Different customer segments
    "Age": [35, 50, 28, 60]  # Younger vs. older customers
})

print("\n📌 Sample Data for Prediction:\n", sample_data)

# Get probability scores instead of direct predictions
probabilities = best_rf.predict_proba(sample_data)[:, 1]  # Probability of being a respondent

# Adjust decision threshold (default is 0.5, now using 0.4)
threshold = 0.4
predictions = (probabilities >= threshold).astype(int)

# Store results
sample_data["Predicted_Response"] = predictions
sample_data["Response_Probability"] = probabilities

print("\n✅ Prediction Results (Threshold = 0.4):\n", sample_data)
