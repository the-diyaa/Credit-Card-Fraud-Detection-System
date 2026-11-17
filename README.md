
# Credit Card Fraud Detection and Risk Analysis System

## Project Overview

This project uses Machine Learning to identify fraudulent
credit card transactions.

The system uses the Random Forest classification algorithm
to distinguish between genuine and fraudulent transactions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Dataset

European Credit Card Fraud Detection Dataset.

## Main Features

1. Data preprocessing
2. Machine learning model training
3. Model evaluation
4. Fraud transaction prediction
5. Streamlit dashboard
6. CSV prediction report

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Preprocess the dataset

python src/data_preprocessing.py

### 3. Train the model

python src/train_model.py

### 4. Evaluate the model

python src/evaluate_model.py

### 5. Launch the application

streamlit run app/app.py

## Output

The application predicts whether uploaded transactions
are genuine or potentially fraudulent.
