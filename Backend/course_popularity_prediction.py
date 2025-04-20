import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# 📌 Load dataset
df = pd.read_csv("backend/data/course_data_modified.csv")

# 📌 Feature engineering - Define a "popularity score"
df['popularity_score'] = (df['num_subscribers'] * 0.5) + (df['num_reviews'] * 0.3) + (df['rating'] * 10)

# 📌 Select features for training
features = ['num_subscribers', 'num_reviews', 'rating', 'price', 'num_lectures', 'content_duration']

# 📌 Convert categorical columns (like 'level', 'subject') into numerical values
encoder = LabelEncoder()
df['level_encoded'] = encoder.fit_transform(df['level'])
df['subject_encoded'] = encoder.fit_transform(df['subject'])

features += ['level_encoded', 'subject_encoded']

# 📌 Define input (X) and output (y)
X = df[features]
y = df['popularity_score']

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Predict future popularity scores
y_pred = model.predict(X_test)

# 📌 Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# 📌 Predict popularity for new courses (Next 2 years projection)
df['predicted_popularity'] = model.predict(X)

import joblib
model_path = "backend/models/course_popularity_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved at {model_path}")
