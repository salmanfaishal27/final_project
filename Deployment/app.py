import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


# Load All Files

with open('model.pkl', 'rb') as file_model:
  model = pickle.load(file_model)

with open('model_scaler.pkl', 'rb') as file_model_scaler:
  scaler = pickle.load(file_model_scaler)



def run():
    st.title('Aria Predict')

    # Membuat form
    with st.form(key='form_parameters'):
        
        v1= st.number_input('v1', min_value=0.0, max_value=None, value= 525.333333)
        v2= st.number_input('v2', min_value=0.0, max_value=None, value= 195.066667)
        v3= st.number_input('v3', min_value=0.0, max_value=None, value= 591.866667)
        v4= st.number_input('v4', min_value=0.0, max_value=None, value= 382.733333)
        v5= st.number_input('v5', min_value=0.0, max_value=None, value= 478.600000)
        v6= st.number_input('v6', min_value=0.0, max_value=None, value= 150.466667)
        v7= st.number_input('v7', min_value=0.0, max_value=None, value= 677.200000)
        v8= st.number_input('v8', min_value=0.0, max_value=None, value= 4850.600000)
        

        submitted = st.form_submit_button('Predict')

    data_inf = {
    'v1': v1,
    'v2': v2,
    'v3': v3,
    'v4': v4,
    'v5': v5,
    'v6': v6,
    'v7': v7,
    'v8': v8,
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    data_inf_scaled = scaler.transform(data_inf)
    

    if submitted:       
        y_pred_inf = model.predict(data_inf_scaled)
        st.write('Predicted :', y_pred_inf[0])

      
if __name__ == '__main__':
    run()