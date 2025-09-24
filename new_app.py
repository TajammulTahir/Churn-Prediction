import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the Traned Model, Scaler, Label, Onehot, Pickle
model = tf.keras.models.load_model('model.keras', compile = False)

# save to the native Keras format (recommended)
m.save("model.keras", save_format="keras")
print("Saved model.keras (native Keras format).")

with open('label_encoder_gender','rb') as file:
    label_encoding_gender= pickle.load(file)

with open('onehot_encoding','rb') as file:
    onehot_encoding= pickle.load(file)

with open('scaler', 'rb') as file:
    scaler= pickle.load(file)

st.title("Customer Churn Prediction")

# User Input
credit_score = st.number_input('CreditScore')  
geography = st.selectbox('Geography', onehot_encoding.categories_[0])
gender = st.selectbox('Gender',label_encoding_gender.classes_)
age = st.slider('Age', 18,92)
tenure = st.slider('Tenure', 0,10)
balance = st.number_input("Balance")
numofproducts= st.slider("NumOfProducts",1,4)
hascrcard = st.selectbox("HasCrCard",[0,1])
isactivemember = st.selectbox("IsActiveMember",[0,1])
estimatedsalary = st.number_input("EstimatedSalary")

# Prepare the Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score], 
    "Gender": [label_encoding_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numofproducts],
    'HasCrCard': [hascrcard],
    'IsActiveMember': [isactivemember],
    'EstimatedSalary': [estimatedsalary],
})

# OneHot Encoding on Geography
geo_df = pd.DataFrame({'Geography': [geography]})
encoded_geo = onehot_encoding.transform(geo_df).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns=onehot_encoding.get_feature_names_out())

# Concat encoded_geo_df into input data
input_data = pd.concat([input_data.reset_index(drop = True), encoded_geo_df], axis = 1)

# Scale the values
input_data_scales = scaler.transform(input_data)

# Prediction Churn
prediction = model.predict(input_data_scales)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The customer is likely to Churn.")
else:
    st.write("The customer is Not likely to Churn.")