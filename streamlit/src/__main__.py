import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(page_title="Dashboard Sber")
st.title('Dashboard Sber')

tab1, tab2 = st.tabs(["Dashboard", "Prediction"])

with tab1:
   st.title("Построение дашбордов")
   st.header('Some header')
   uploaded_file = st.file_uploader("Выберите файл", key="dashboard")
   dataframe = None
   if uploaded_file is not None:
      dataframe = pd.read_csv(uploaded_file)
      st.write(dataframe)

   if dataframe:
      st.header('My header')
      st.text("")
      st.image('./header.png')

      st.header('My header')
      st.text("")
      st.line_chart(np.random.randn(30, 3))

      st.header('My header')
      st.text("")
      st.line_chart(np.random.randn(30, 3))

with tab2:
   st.title("Предсказание значений")

   uploaded_file = st.file_uploader("Выберите файл", key="predict")
   dataframe = None
   if uploaded_file is not None:
      dataframe = pd.read_csv(uploaded_file)
      st.write(dataframe)
   
   if dataframe:
      st.dataframe(dataframe)
