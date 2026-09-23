# 🎓 Student Performance Predictor

## Project Overview
This project is a complete, working machine learning portfolio project built for the Women Innovating in Cloud Africa (WIICA) program. It features an end-to-end data pipeline, model training, and a live Streamlit web dashboard.

## Problem Statement & Objective
Educators and parents often struggle to identify which habits most strongly impact a student's grades. The objective is to build, evaluate, and deploy a machine learning regression model that predicts a student’s final academic performance using factors such as study habits, attendance, sleep, parental involvement, extracurricular activities, and demographic/academic indicators.

## Dataset & Features
*   **Dataset:** Student Performance Factors.
*   **Features Used:** Hours Studied, Attendance Percentage, Sleep Hours, Parental Involvement, Tutoring Sessions, Access to Resources, and various demographic indicators.
*   **Target Variable:** Final Exam Score (0-100).

## Machine Learning Approach & Data Preprocessing
*   **Preprocessing:** Handled missing values, encoded categorical features, and scaled numerical features. Data was split into 80% training and 20% testing sets.
*   **Models Evaluated:** We trained and compared both a Random Forest Regressor and a Linear Regression model.

## Evaluation Metrics & Results
The models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score.
*   **Linear Regression (Winner):** R² = 0.7314 | MAE = 0.49 | RMSE = 2.04
*   **Random Forest:** R² = 0.6175 | MAE = 1.24 | RMSE = 2.44
*   *Linear Regression was selected as the final model due to its superior ability to capture the straightforward linear relationships in the dataset.*

## Streamlit Application
* https://student-performance-predictor-9ebcq7kuvt5moka4esqfhb.streamlit.app/
* The interactive web application (`app.py`) allows users to input student parameters via dynamic sidebar sliders and dropdowns. It provides:
*   Real-time final score predictions.
*   Actionable performance feedback (High Performance, Average, At-Risk).
*   An interactive Plotly bar chart displaying the key study habits driving the score prediction.

## Project Structure
```text
student-performance-predictor/
├── app.py
├── student_model.pkl
├── student_scaler.pkl
├── Student_Performance.ipynb
├── requirements.txt
└── README.md
``` []

## How to Run Locally
1. Clone this repository.
2. Open your command prompt/terminal and navigate to the project folder.
3. Install the required libraries by running: `pip install -r requirements.txt`
4. Launch the app by running: `streamlit run app.py`

## Deployment
This app is designed to be deployed via Streamlit Community Cloud directly from this GitHub repository.

## Limitations & Future Improvements
*   **Limitations:** The model assumes linear relationships and is trained on a specific dataset; it does not guarantee a student's actual final grade. Feature importance indicates predictive usefulness, not strict causation.
*   **Future Improvements:** Incorporating more diverse, real-world data and experimenting with advanced ensemble models (like XGBoost).

## Author
[Naa Norley Norteye] - WIICA Capstone Project
