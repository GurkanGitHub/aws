import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Churn Classification")


model=pickle.load(open("logmodel.pkl", "rb")) 

tenure = st.sidebar.slider("Duration of stay (tenure):",0,2, step=1)
MonthlyCharges = st.sidebar.slider("monthly charge of customer: ", 0,100, step=5)
TotalCharges = st.sidebar.slider("total charge of customer:", 0,5000,step=10)


my_dict = {"tenure":tenure,
           "MonthlyCharges":MonthlyCharges,
           "TotalCharges":TotalCharges}
df = pd.DataFrame.from_dict([my_dict])

st.table(df)

classification = model.predict(df)
st.write(classification)


# def single_customer():
#     my_dict = {"tenure": tenure
#                ,"MonthlyCharges":MonthlyCharges
#                ,"TotalCharges":TotalCharges}
#     df_sample = pd.DataFrame.from_dict([my_dict])
#     return df_sample
# df = single_customer()
# st.table(df)
