Liver Disease Multi Class Prediction System
Project Overview

This project focuses on building a multi class liver disease prediction system using machine learning techniques. The model analyzes clinical laboratory parameters and predicts the disease category based on learned patterns from the dataset.

The project demonstrates a complete end to end machine learning workflow including data preprocessing, model training, evaluation, model selection, and deployment using Streamlit.

Problem Statement

Early detection of liver disease is important for timely medical intervention. This system uses patient laboratory data to classify liver disease categories and assist in prediction based on statistical learning patterns.

Dataset Features

The model was trained using the following features:

Age

Sex

Albumin

Alkaline Phosphatase

Alanine Aminotransferase

Aspartate Aminotransferase

Bilirubin

Cholinesterase

Cholesterol

Creatinina

Gamma Glutamyl Transferase

Protein

Target variable: Disease Category

Machine Learning Workflow

Data Cleaning

Removed duplicates

Handled missing values

Feature Engineering

Encoded categorical variables

Applied feature scaling

Model Training
The following models were trained and evaluated:

Logistic Regression

Random Forest

Support Vector Machine

K Nearest Neighbors

Model Selection
Logistic Regression achieved the highest accuracy of 96.62 percent and was selected as the final model.

Model Saving

model.pkl

scaler.pkl

label_encoder.pkl

Deployment
The final model was deployed using Streamlit, allowing users to enter patient data and receive real time predictions along with class probabilities.

How to Run Locally

Clone the repository:

git clone https://github.com/Tejas128-D/Liver-Disease-Prediction-ML.git

Navigate to the project folder:

cd Liver-Disease-Prediction-ML

Install dependencies:

pip install -r requirements.txt

Run the app:

python -m streamlit run app.py

Technologies Used

Python

Scikit Learn

Pandas

NumPy

Streamlit

Joblib

Key Highlights

End to end ML pipeline implementation

Proper preprocessing and feature scaling

Model comparison and evaluation

Label decoding for deployment

Real time web application deployment

Future Improvements

Improve class imbalance handling

Add input validation for production usage

Implement model monitoring and retraining strategy

Enhance UI with better visualization

Author

Tejas R
