import pandas as pd

# Load dataset
df = pd.read_csv("backend/processed_courses.csv")


# Display basic info
print(df.info())
print(df.head())
# Fill missing numerical values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing categorical values with mode (most frequent value)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify
print(df.isnull().sum())
# Convert date columns to datetime format (if applicable)
if 'published_date' in df.columns:
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# Convert price to numeric (if applicable)
if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Verify changes
print(df.dtypes)

df.drop_duplicates(inplace=True)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['price', 'num_subscribers']] = scaler.fit_transform(df[['price', 'num_subscribers']])


df.to_csv("cleaned_courses.csv", index=False)
print("Processing completed and saved as cleaned_courses.csv")
