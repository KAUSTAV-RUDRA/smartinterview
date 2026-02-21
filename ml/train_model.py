import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure we read the file relative to the script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.csv")
data = pd.read_csv(data_path)

X = data[['experience', 'skills', 'quiz']]
y = data['selected']

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X, y)

# Save the model relative to the script's directory
model_path = os.path.join(current_dir, "..", "model.pkl")
joblib.dump(model, model_path)

print(f"RandomForestClassifier model trained on {len(data)} samples and saved to {model_path}")