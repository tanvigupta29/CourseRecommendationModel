import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\Nitin Gupta\Desktop\courses rec\data\udemy_courses.csv")


# Fill NaN values only for numeric columns
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

# Save the processed data (optional)
df.to_csv("processed_courses.csv", index=False)
