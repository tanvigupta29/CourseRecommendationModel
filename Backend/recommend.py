import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader, dump

# Define model path
model_path = "backend/models/course_recommendation_model.pkl"

# Load the trained model
try:
    print(" Attempting to load model...")
    model_data = dump.load(model_path)
    
    if isinstance(model_data, tuple):  
        _, model = model_data  # Extract the trained SVD model
    else:
        model = model_data  

    if model is None:
        print(" Model is None after loading!")
    else:
        print(f" Model loaded successfully! Type: {type(model)}")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

# Load dataset
df = pd.read_csv("backend/data/course_data_modified.csv")  

print("Columns in dataset:", df.columns)
print(" Dataset Preview:")
print(df.head())

# Prepare dataset for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)

def recommend_courses_for_user(user_id, num_recommendations=5):
    """Recommend top N courses for a given user based on predicted ratings."""
    if model is None:
        print(" Model is not loaded. Cannot make recommendations.")
        return []

    all_courses = df["course_id"].unique()
    predictions = [(course, model.predict(user_id, course).est) for course in all_courses]

    # Sort courses by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_courses = [course for course, _ in predictions[:num_recommendations]]

    return top_courses

# Example usage (Test the function)
if __name__ == "__main__":
    test_user_id = 1  # Example user_id
    recommendations = recommend_courses_for_user(test_user_id)
    print(f" Recommended courses for user {test_user_id}: {recommendations}")

def recommend_courses_for_user(user_id, available_courses, num_recommendations=5):
    """Recommend courses based on the trained model."""
    predictions = [(course, model.predict(user_id, course).est) for course in available_courses]

    # Sort courses by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_courses = [course for course, _ in predictions[:num_recommendations]]

    return top_courses
