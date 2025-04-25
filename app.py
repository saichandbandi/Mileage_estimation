###### Libraries #######

# Base Libraries
import pandas as pd
import numpy as np

# Deployment Library
import streamlit as st

# Model Pickled File Library
import joblib

############# Data File ###########

data = pd.read_excel("AutompgEstimation (6X).xlsx")

data = data.dropna(axis=0).reset_index(drop=True)

########### Loading Trained Model Files ########
model = joblib.load("autompg_rfreg.pkl")


########## UI Code #############

# Ref: https://docs.streamlit.io/library/api-reference

# Title
st.header("Estimation of mileage for the Given vehicles Details:")

# Description
st.write("""Built a Predictive model in Machine Learning to estimate the mileage of a given vehicle can get.
         Sample Data taken as below shown.
""")
# Data Monitoring

# Data Display
st.dataframe(data.head())
st.write("From the above data , mileage is the prediction variable")

###### Taking User Inputs #########
st.subheader("Enter Below Details to Get the Estimation Mileage:")

col1, col2, col3 = st.columns(3) # value inside brace defines the number of splits
col4, col5, col6 = st.columns(3)


with col1:
    carname= st.selectbox("Enter the vehicle name:",data.carname.unique())
    st.write(carname)

with col2:
    modelyear = st.selectbox("model year(range from 1970 to 1982):",data.modelyear.unique())
    st.write(modelyear)

with col3:
    horsepower = st.number_input("horse power:")
    st.write(horsepower)

with col4:
    cylinders = st.number_input("Enter Number of cylinders:")
    st.write(cylinders)

with col5:
    origin = st.selectbox("Enter origin:", data.origin.unique())
    st.write(origin)

with col6:
    accelration = st.selectbox("accelration:", data.accelration.unique())
    st.write(accelration)

###### Predictions #########

if st.button("Estimate"):
    st.write("Data Given:")
    values = [carname, modelyear, horsepower, cylinders, origin, accelration]
    record =  pd.DataFrame([values],
                           columns = ['carname', 'modelyear',
                                      'horsepower', 'cylinders',
                                      'origin', 'accelration'])
    
    st.dataframe(record)
    mileage = round(model.predict(record)[0],2)
    mileage = str(mileage)+ 'kmpl'
    st.subheader("Estimated Mileage:")
    st.subheader(mileage)