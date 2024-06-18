# Health_care_using_ML
Overview
This project aims to predict the occurrence of various diseases, including diabetes, heart disease, Parkinson's disease, and COVID-19, using machine learning algorithms. The project provides a user-friendly interface for inputting patient data and obtaining predictions for each disease.

Models Used
The project utilizes pre-trained machine learning models for disease prediction:

Diabetes Prediction: Logistic Regression model trained on patient data to predict the likelihood of diabetes based on demographic and medical information.

Heart Disease Prediction: Logistic Regression model trained on a dataset containing various heart disease risk factors to predict the presence of heart disease.

Parkinson's Disease Prediction: Support Vector Machine (SVM) classifier trained on features extracted from voice recordings of patients to predict Parkinson's disease.

COVID-19 Prediction: Convolutional Neural Network (CNN) model trained on chest X-ray images to detect the presence of COVID-19 pneumonia.

Usage
To use the disease prediction application:

Select the disease prediction option from the sidebar menu (Diabetes Prediction, Heart Disease Prediction, Parkinson's Prediction, COVID Prediction).
Enter the required input data for the selected disease prediction.
Click the "Predict" button to obtain the prediction result.
The application will display the prediction result indicating whether the patient is likely to have the disease or not.

Dependencies
  Python (>=3.6)
  TensorFlow (for COVID-19 prediction)
  Streamlit
  NumPy
  Pandas
  scikit-learn

Installation
  Clone or download the project repository:
  bash
  git clone https://github.com/your-username/disease-prediction.git
  Install the required dependencies using pip:
  pip install -r requirements.txt

Usage
  Run the Streamlit application: streamlit run disease_prediction_app.py

Access the application in your web browser and navigate to the desired disease prediction page.
Enter the required input data and click the "Predict" button to obtain the prediction result.

Covid model is not accurate currently working on that 

Contributing
Contributions to this project are welcome.
