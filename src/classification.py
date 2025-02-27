import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("data/clustered_data.csv")

# Use only the selected important features
selected_features = ['Total_Spending', 'Recency', 'Cluster', 'Age']
X = df[selected_features]
y = df['Response']  # 1 = responded, 0 = did not respond

# Split dataset into training and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

### **📌 Fine-Tuning Random Forest with Expanded GridSearch**
print("\n📌 Fine-Tuning Random Forest with Expanded Hyperparameters...")
param_grid_rf = {
    "n_estimators": [100, 200, 300, 400],  # More trees
    "max_depth": [5, 10, 15],  # Controlling depth
    "min_samples_split": [2, 5, 10]  # Preventing overfitting
}

grid_rf = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42), 
                       param_grid_rf, scoring="precision", cv=5, n_jobs=-1, verbose=2)
grid_rf.fit(X_train, y_train)

# Get the best model from tuning
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate the best fine-tuned model
print("\n✅ Best Fine-Tuned Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_rf))
print("📌 Best Fine-Tuned Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Print best hyperparameters
print("\n✅ Best Hyperparameters Found:", grid_rf.best_params_)

# Save the best model
model_path = "models/random_forest_best.pkl"
joblib.dump(best_rf, model_path)

print(f"✅ Model saved successfully at {model_path}")

