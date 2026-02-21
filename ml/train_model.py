import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import joblib

# Ensure we read the file relative to the script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.csv")
data = pd.read_csv(data_path)

X = data[['experience', 'skills', 'quiz']]
y = data['selected']

model = LogisticRegression()
model.fit(X, y)

# Save the model relative to the script's directory
model_path = os.path.join(current_dir, "..", "model.pkl")
joblib.dump(model, model_path)

print("Model trained")