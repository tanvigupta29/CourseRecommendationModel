from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load the dataset and trained AI model
df = pd.read_csv("backend/data/course_data_modified.csv")  # Replace with actual dataset
model = joblib.load("backend/models/course_recommendation_model.pkl")  # Load AI model

# Store conversation state (In-memory session for simplicity)
user_sessions = {}

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_id = data.get("user_id", "default")  # Unique user session
    user_input = data.get("message", "").strip().lower()

    # Initialize user session if not exists
    if user_id not in user_sessions:
        user_sessions[user_id] = {"step": 0, "filters": {}}

    session = user_sessions[user_id]

    # Step 1: Ask course type (Paid/Free)
    if session["step"] == 0:
        session["step"] += 1
        return jsonify({"reply": "Would you like a Paid or Free course?", "options": ["Paid", "Free"]})

    # Step 2: Save course type & Ask for price range
    elif session["step"] == 1:
        session["filters"]["type"] = user_input.capitalize()
        session["step"] += 1
        return jsonify({"reply": "What is your maximum price range?", "options": ["10", "50", "100", "500"]})

    # Step 3: Save price & Ask for subject
    elif session["step"] == 2:
        session["filters"]["price_range"] = int(user_input)
        session["step"] += 1
        return jsonify({"reply": "What subject are you interested in?", "options": ["Data Science", "Web Development", "AI", "Business"]})

    # Step 4: Save subject & Ask duration
    elif session["step"] == 3:
        session["filters"]["subject"] = user_input
        session["step"] += 1
        return jsonify({"reply": "What duration do you prefer?", "options": ["4 weeks", "8 weeks", "12 weeks"]})

    # Step 5: Recommend courses
    elif session["step"] == 4:
        session["filters"]["duration"] = user_input
        recommendations = get_course_recommendations(session["filters"])

        if not recommendations:
            return jsonify({"reply": "Sorry, no courses match your criteria. Try different filters!"})
        
        # Reset session
        user_sessions[user_id] = {"step": 0, "filters": {}}

        return jsonify({"reply": "Here are the top 5 recommended courses:", "courses": recommendations})

    return jsonify({"reply": "I didn't understand. Please follow the chatbot flow."})


def get_course_recommendations(filters):
    """Filters courses & uses AI model to predict the top 5 best matches."""
    filtered_df = df.copy()
    if filters["subject"]:
        filtered_df = filtered_df[filtered_df["subject"] == filters["subject"]]
    if filters["price_range"]:
        filtered_df = filtered_df[filtered_df["price"] <= filters["price_range"]]
    if filters["duration"]:
        filtered_df = filtered_df[filtered_df["duration"] == filters["duration"]]
    if filters["type"]:
        filtered_df = filtered_df[filtered_df["type"] == filters["type"]]

    if filtered_df.empty:
        return []  # No matching courses

    # Use AI model to rank courses
    predictions = model.predict(filtered_df[["price", "duration_encoded", "subject_encoded"]])
    filtered_df["score"] = predictions

    # Get top 5 courses
    top_courses = filtered_df.sort_values(by="score", ascending=False).head(5)
    return top_courses[["name", "subject", "price", "duration", "type"]].to_dict(orient="records")


if __name__ == '__main__':
    app.run(debug=True)
