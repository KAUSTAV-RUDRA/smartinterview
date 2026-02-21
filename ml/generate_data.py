import pandas as pd
import numpy as np
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 1000 samples
n_samples = 1000

# Generate synthetic features
# Experience between 0 and 20 years
experience = np.random.randint(0, 21, n_samples)
# Skills count between 1 and 20
skills = np.random.randint(1, 21, n_samples)
# Quiz score between 30 and 100
quiz = np.random.randint(30, 101, n_samples)

# Let's create a realistic probability of being selected based on the features
# We weight the features: quiz score and skills are very important, experience is also good
# Normalize the scale points arbitrarily
score = (experience * 2) + (skills * 3) + (quiz * 0.8)

# Add some gaussian noise to simulate real-world fuzziness in hiring
score += np.random.normal(0, 15, n_samples)

# Let's set the threshold so about 35% of candidates are selected
threshold = np.percentile(score, 65)
selected = (score >= threshold).astype(int)

df = pd.DataFrame({
    'experience': experience,
    'skills': skills,
    'quiz': quiz,
    'selected': selected
})

# Save to data.csv
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.csv")
df.to_csv(data_path, index=False)

print(f"Generated a new synthetic dataset with {len(df)} rows and stored it in data.csv")
