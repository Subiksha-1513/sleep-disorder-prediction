import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page setup
st.set_page_config(page_title="Sleep Disorder Predictor 😴", layout="wide")

# Title
st.title("💤 Sleep Disorder Prediction App")
st.write("Predict your possible sleep disorder based on your lifestyle and health details.")

# Load dataset
df = pd.read_csv("sleep_health.csv")

# Encode categorical columns
le = LabelEncoder()
for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df[['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]
y = df['Sleep Disorder']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧍‍♀️ Personal Information")
    gender = st.selectbox("Gender", df['Gender'].unique())
    age = st.slider("Age", 18, 80, 25)
    occupation = st.selectbox("Occupation", df['Occupation'].unique())

with col2:
    st.subheader("💪 Health & Lifestyle")
    sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0)
    quality = st.slider("Quality of Sleep (1-10)", 1, 10, 6)
    activity = st.number_input("Physical Activity Level", 0, 100, 40)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    heart_rate = st.number_input("Heart Rate", 40, 120, 70)
    steps = st.number_input("Daily Steps", 1000, 20000, 4000)

# Encode gender input
gender_encoded = le.fit_transform(df['Gender'])
gender_map = dict(zip(df['Gender'], gender_encoded))
gender_val = gender_map[gender]

# Prepare input data
user_data = pd.DataFrame({
    'Gender': [gender_val],
    'Age': [age],
    'Sleep Duration': [sleep_duration],
    'Quality of Sleep': [quality],
    'Physical Activity Level': [activity],
    'Stress Level': [stress],
    'Heart Rate': [heart_rate],
    'Daily Steps': [steps]
})

# Prediction
st.markdown("---")
if st.button("🔮 Predict Sleep Disorder"):
    prediction = model.predict(user_data)
    result = le.inverse_transform(prediction)[0]

    st.success(f"🩺 **Predicted Sleep Disorder:** {result}")

    if result == "Insomnia":
        st.info("💡 Try improving your bedtime habits and reduce stress for better sleep.")
    elif result == "Sleep Apnea":
        st.warning("⚠️ You might need to consult a sleep specialist for further evaluation.")
    else:
        st.balloons()
        st.success("🎉 You appear to have **no sleep disorder!** Keep up the healthy routine!")

# Optional Dataset Preview
with st.expander("🧾 View Dataset Sample"):
    st.dataframe(df.head(10))
