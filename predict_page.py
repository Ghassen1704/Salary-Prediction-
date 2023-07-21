import streamlit as st
import pickle 
import numpy as np 

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data

data=load_model()
regressor_loaded=data["model"]
l_country=data["l_country"]
l_education=data["l_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""" )
    
    countries=(
    "United States of America",
    "India",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden",
)
    education=(
    "Less than a Bachelor’s degree",
    "Bachelor’s degree",
    "Master’s degree",
    "Professional degree",

)
    country=st.selectbox("Country",countries)
    education=st.selectbox("Education Level",education)
    experience=st.slider("Years of Experience",0,50,3)
    ok=st.button("Calculate Salary")
    if ok:
        x=np.array([[country,education,experience]])
        x[:,0]=l_country.transform(x[:,0])
        x[:,1]=l_education.transform(x[:,1])
        x=x.astype(float)
        salary=regressor_loaded.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
