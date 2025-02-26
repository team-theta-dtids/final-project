import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the saved model and preprocessing pipeline
model_filename = "output/xgboost_model.pkl"
with open(model_filename, 'rb') as file:
    pipeline_xgb = pickle.load(file)

# Define the decision threshold
THRESHOLD = 0.75

# Expected features from X_train (after feature engineering)
expected_features = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome',
    'nr.employed', 'age_group_middle-aged', 'age_group_senior', 'economic_indicator'
]

# Function to preprocess input data
def preprocess_input_data(input_data):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Remove extra features (if any)
    input_df = input_df.reindex(columns=expected_features)
    
    # Add missing features with default values
    for feature in expected_features:
        if feature not in input_df.columns:
            if feature.startswith('age_group'):  # Binary feature
                input_df[feature] = 0
            elif feature == 'economic_indicator':  # Numerical feature
                input_df[feature] = 0
            else:  # Categorical feature
                input_df[feature] = 'unknown'
    
    # Recreate engineered features if possible
    if 'age' in input_df.columns:
        input_df['age_group_middle-aged'] = ((input_df['age'] >= 30) & (input_df['age'] < 50)).astype(int)
        input_df['age_group_senior'] = (input_df['age'] >= 50).astype(int)
    
    if all(col in input_df.columns for col in ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']):
        input_df['economic_indicator'] = (
            input_df['emp.var.rate'] * 0.4 +
            input_df['cons.price.idx'] * 0.3 +
            input_df['cons.conf.idx'] * 0.2 +
            input_df['euribor3m'] * 0.1
        )
    
    # Drop raw features used for feature engineering
    raw_features_to_drop = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']
    input_df = input_df.drop(columns=[col for col in raw_features_to_drop if col in input_df.columns], errors='ignore')
    
    # Reorder columns to match the order in X_train
    input_df = input_df[expected_features]
    
    return input_df

# Function to make single predictions
def predict_single(input_data):
    # Preprocess the input data
    processed_data = preprocess_input_data(input_data)
    
    # Get predicted probabilities
    proba = pipeline_xgb.predict_proba(processed_data)[:, 1]
    # Apply threshold
    prediction = (proba >= THRESHOLD).astype(int)
    return prediction[0], proba[0]

# Function to make batch predictions
def predict_batch(batch_data):
    # Remove extra features
    batch_data = batch_data.reindex(columns=expected_features)
    
    # Add missing features with default values
    for feature in expected_features:
        if feature not in batch_data.columns:
            if feature.startswith('age_group'):  # Binary feature
                batch_data[feature] = 0
            elif feature == 'economic_indicator':  # Numerical feature
                batch_data[feature] = 0
            else:  # Categorical feature
                batch_data[feature] = 'unknown'
    
    # Recreate engineered features if possible
    if 'age' in batch_data.columns:
        batch_data['age_group_middle-aged'] = ((batch_data['age'] >= 30) & (batch_data['age'] < 50)).astype(int)
        batch_data['age_group_senior'] = (batch_data['age'] >= 50).astype(int)
    
    if all(col in batch_data.columns for col in ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']):
        batch_data['economic_indicator'] = (
            batch_data['emp.var.rate'] * 0.4 +
            batch_data['cons.price.idx'] * 0.3 +
            batch_data['cons.conf.idx'] * 0.2 +
            batch_data['euribor3m'] * 0.1
        )
    
    # Drop raw features used for feature engineering
    raw_features_to_drop = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']
    batch_data = batch_data.drop(columns=[col for col in raw_features_to_drop if col in batch_data.columns], errors='ignore')
    
    # Reorder columns to match the order in X_train
    batch_data = batch_data[expected_features]
    
    # Get predicted probabilities
    proba = pipeline_xgb.predict_proba(batch_data)[:, 1]
    # Apply threshold
    predictions = (proba >= THRESHOLD).astype(int)
    return predictions, proba

# Streamlit App
st.title("Customer Subscription Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Performance", "Single Prediction", "Batch Prediction"])

# Page 1: Model Performance Visualization
if page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Load preprocessed test data
    try:
        X_test = pd.read_csv("output/X_test.csv")  # Replace with your test dataset path
        y_test = pd.read_csv("output/y_test.csv")['y']  # Replace with your test labels path
    except FileNotFoundError:
        st.error("Test data files (X_test.csv and y_test.csv) not found. Please ensure they are in the correct directory.")
        st.stop()
    
    # Encode y_test into numerical values
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    
    # Make predictions on the test set
    y_pred_proba = pipeline_xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)
    
    # Display classification report
    st.subheader("Classification Report")
    report = classification_report(y_test_encoded, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test_encoded, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"]).plot(ax=ax)
    st.pyplot(fig)

# Page 2: Single Prediction
elif page == "Single Prediction":
    st.header("Single Customer Prediction")
    
    # Input fields for user
    st.subheader("Enter Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                                'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
    marital = st.selectbox("Marital Status", ['divorced', 'married', 'single'])
    education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                           'illiterate', 'professional.course', 'university.degree'])
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
    month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox("Last Contact Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    campaign = st.number_input("Number of Contacts During This Campaign", min_value=0, value=1)
    pdays = st.number_input("Days Passed After Last Contact", min_value=0, value=999)
    previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, value=0)
    poutcome = st.selectbox("Outcome of Previous Campaign", ['failure', 'nonexistent', 'success'])
    nr_employed = st.number_input("Number of Employees", value=5000.0)
    emp_var_rate = st.number_input("Employment Variation Rate", value=-1.8)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.number_input("Euribor 3-Month Rate", value=1.0)
    
    # Prepare input data as a dictionary
    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'nr.employed': nr_employed,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m
    }
    
    # Predict button
    if st.button("Predict"):
        prediction, probability = predict_single(input_data)
        if prediction == 1:
            st.success(f"The customer is likely to subscribe (Probability: {probability:.2f})")
        else:
            st.error(f"The customer is unlikely to subscribe (Probability: {probability:.2f})")

# Page 3: Batch Prediction
elif page == "Batch Prediction":
    st.header("Batch Prediction")
    
    # Instructions for uploading a CSV file
    st.info("""
    **Instructions for Uploading a CSV File:**
    
    To ensure successful batch predictions, the uploaded CSV file must meet the following requirements:
    
    1. **Expected Features:**  
       The file must contain the following 18 features (columns) in the exact order:
       ```plaintext
       ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome',
        'nr.employed', 'age_group_middle-aged', 'age_group_senior', 'economic_indicator']
       ```
       
    2. **Engineered Features:**  
       - `age_group_middle-aged`: Binary feature indicating if the customer is middle-aged (30–49 years).
       - `age_group_senior`: Binary feature indicating if the customer is a senior (50+ years).
       - `economic_indicator`: A numerical feature derived from economic variables (`emp.var.rate`, `cons.price.idx`, etc.).
       
    3. **Raw Features:**  
       Raw features like `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, and `euribor3m` should **not** be included.
       
    4. **Data Cleaning:**  
       - Ensure that all missing values are handled appropriately.
       - Categorical features should use the same categories as the training data.
       
    5. **Example Structure:**  
       Here’s an example of how the first few rows of your CSV file should look:
       ```csv
       age,job,marital,education,default,housing,loan,contact,month,day_of_week,campaign,pdays,previous,poutcome,nr.employed,age_group_middle-aged,age_group_senior,economic_indicator
       30,admin.,married,high.school,no,no,no,cellular,may,mon,1,999,0,nonexistent,5000.0,1,0,1.23
       45,technician,single,university.degree,yes,yes,no,telephone,jun,tue,2,100,1,failure,5100.0,1,0,0.98
       ```
       
    If your file does not meet these requirements, the app may fail to process it. You can preprocess your data using the same steps applied to the training data (`X_train`) before uploading.
    """)
    
    # File uploader for batch prediction
    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded file
        batch_data = pd.read_csv(uploaded_file)
        
        # Ensure the uploaded data has the same preprocessing steps
        try:
            predictions, probabilities = predict_batch(batch_data)
            
            # Add predictions and probabilities to the DataFrame
            batch_data['Prediction'] = predictions
            batch_data['Probability'] = probabilities
            
            # Decode predictions back to 'no'/'yes' for better readability
            label_encoder = LabelEncoder()
            batch_data['Prediction'] = label_encoder.fit_transform(batch_data['Prediction'])
            batch_data['Prediction'] = label_encoder.inverse_transform(batch_data['Prediction'])
            
            # Display results
            st.subheader("Batch Prediction Results")
            st.dataframe(batch_data)
            
            # Download results as CSV
            st.download_button(
                label="Download Results as CSV",
                data=batch_data.to_csv(index=False).encode('utf-8'),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {str(e)}")