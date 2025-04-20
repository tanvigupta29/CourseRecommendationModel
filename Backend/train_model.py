import pandas as pd

# Load dataset
df = pd.read_csv("backend/data/cleaned_courses.csv")  

# Ensure 'rating' column exists
if 'rating' not in df.columns:
    df['rating'] = df['num_subscribers'] / df['num_subscribers'].max() * 5  

# Ensure 'user_id' column exists
if 'user_id' not in df.columns:
    df['user_id'] = range(1, len(df) + 1)

# Save the modified dataset to verify changes
df.to_csv("course_data_modified.csv", index=False)

# Print columns to verify
print("Updated Dataset Columns:", df.columns)
print(df.head())  # Preview dataset
