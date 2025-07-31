#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


model = joblib.load('my_model.pkl')


# In[3]:


st.title("Titanic Survival Predictor")


# Pclass	Sex	Age	SibSp	Fare	Embarked_S	Tickets

# In[4]:


def user_input():
    Pclass = st.sidebar.selectbox("Please Select Your Class A-1,B-2,C-3",[0,1,2,3])
    Sex = st.sidebar.selectbox("Please Select Your Sex Male-1,Female-2",[0,1,2])
    Age = st.sidebar.slider("Select Your Age",0,100)
    SibSp = st.sidebar.selectbox("Select Your Journy type Alone-0,Two members -2,Morethan that-3",[0,1,2])
    Fare = st.sidebar.slider("Select Your Fare Expensive",0,600)
    Embarked_S = st.sidebar.selectbox("Select Your City (S-1,C-2,Q-3)",[0,1,2,3])
    Tickets = st.sidebar.slider("Enter Your Tickets Number Only numbers",0,10)
    dict1={'Pclass':Pclass,'Sex':Sex, 'Age':Age, 'SibSp':SibSp ,'Fare':Fare ,'Embarked_S':Embarked_S ,'Tickets':Tickets}
    return pd.DataFrame(dict1,index=[0])
    
df =user_input()
if st.button("Predict"):
    pred_prob = model.predict_proba(df)
    prdic = model.predict(df)
    st.subheader('Predicted')
    st.write('Survived' if prdic[0] == 1 else 'No more')
    st.subheader('Predicted_proba')
    st.write(pred_prob)


# In[5]:


df


# In[ ]:




