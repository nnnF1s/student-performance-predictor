import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="",
    layout="centered"
)

# 2. Load the Saved Model and Scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('student_model.pkl')
    scaler = joblib.load('student_scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# 3. App Header
st.title("Student Performance Predictor")
st.markdown("Welcome to the WIICA Machine Learning Capstone Project! Use the controls below to enter student habits and predict their final exam score.")
st.markdown("---")

# 4. Sidebar Inputs (User Controls)
st.sidebar.header("Student Parameters")

def user_input_features():
    hours_studied = st.sidebar.slider("Hours Studied (Weekly)", 1, 44, 15)
    attendance = st.sidebar.slider("Attendance (%)", 60, 100, 85)
    sleep_hours = st.sidebar.slider("Sleep Hours (Daily)", 4, 10, 7)
    previous_scores = st.sidebar.slider("Previous Scores", 50, 100, 75)
    tutoring_sessions = st.sidebar.slider("Tutoring Sessions (Monthly)", 0, 8, 2)
    physical_activity = st.sidebar.slider("Physical Activity (Weekly Hours)", 0, 6, 3)
    
    parental_involvement = st.sidebar.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.sidebar.selectbox("Access to Resources", ["Low", "Medium", "High"])
    extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
    motivation_level = st.sidebar.selectbox("Motivation Level", ["Low", "Medium", "High"])
    internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
    family_income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.sidebar.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    school_type = st.sidebar.selectbox("School Type", ["Public", "Private"])
    peer_influence = st.sidebar.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
    learning_disabilities = st.sidebar.selectbox("Learning Disabilities", ["Yes", "No"])
    parental_education = st.sidebar.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    distance_from_home = st.sidebar.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Tutoring_Sessions': tutoring_sessions,
        'Physical_Activity': physical_activity,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Extracurricular_Activities': extracurricular,
        'Motivation_Level': motivation_level,
        'Internet_Access': internet_access,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'School_Type': school_type,
        'Peer_Influence': peer_influence,
        'Learning_Disabilities': learning_disabilities,
        'Parental_Education_Level': parental_education,
        'Distance_from_Home': distance_from_home,
        'Gender': gender
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# 5. Main Dashboard Display
st.subheader("Current Student Profile")
st.write(df_input)

# 6. Prediction Logic
if st.button("Predict Final Exam Score", type="primary"):
    try:
        numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
        categorical_cols = [col for col in df_input.columns if col not in numerical_cols]
        
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
        
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            df_encoded = df_encoded.reindex(columns=expected_cols, fill_value=0)

        prediction = model.predict(df_encoded)[0]
        prediction = max(0, min(100, prediction))

        st.success("Prediction Successful!")
        st.metric(label="Predicted Final Exam Score", value=f"{prediction:.2f} / 100")

        # Performance Feedback Categories
        if prediction >= 75:
            st.info("🌟 **High Performance:** This student is on track to achieve an excellent grade! Keep up the great habits.")
        elif prediction >= 60:
            st.warning("⚠️ **Average Performance:** This student is performing reasonably, but there is room for improvement in attendance or study hours.")
        else:
            st.error("🚨 **At Risk:** This student is projected to score low. Immediate intervention with tutoring and increased attendance is recommended.")

        # --- NEW: Analytics / Plotly Feature Impact Chart ---
        st.markdown("---")
        st.subheader("Key Study Habits Driving Predictions")
        st.caption("This chart displays the top features influencing the Linear Regression model's predictions.")

        if hasattr(model, "coef_") and hasattr(model, "feature_names_in_"):
            coef_data = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Impact': model.coef_
            })
            coef_data['Abs_Impact'] = coef_data['Impact'].abs()
            top_factors = coef_data.sort_values(by='Abs_Impact', ascending=False).head(8)

            fig = px.bar(
                top_factors,
                x='Impact',
                y='Feature',
                orientation='h',
                color='Impact',
                color_continuous_scale='Bluered',
                title="Top Influential Factors on Final Score",
                labels={'Impact': 'Impact (Points Added / Deducted)', 'Feature': 'Habit / Factor'}
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# 7. About the App section
st.markdown("---")
st.markdown("### About the Model")
st.markdown("""
- **Algorithm:** Linear Regression (Chosen over Random Forest due to higher R² and lower MAE).
- **Key Drivers:** Attendance percentage and weekly hours studied have the strongest positive impact on scores.
- **Developer:** Built with Python, scikit-learn, and Streamlit for the WIICA Capstone Project.
""")
