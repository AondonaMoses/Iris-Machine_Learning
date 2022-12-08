# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 01:02:51 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

loaded_model = pickle.load(open("trained_model.sav", 'rb'))



def iris_prediction(input_data):
    
    input_data_as_array = np.asarray(input_data)
    input_data_reshape = input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshape)
    
    print(prediction)
    if (prediction[0] == 0):
        return "Iris Setosa"
    elif (prediction[0] == 1):
        return "Iris Versicolor"
    else:
        return "Iris Virginica"
        
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)
    
    data = {'Sepal_length': sepal_length,
            'Sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    
    features = pd.DataFrame(data, index=[0])
    
    return features

def main():
    st.write("""
             # Iris Flower Prediction Web APP
             
             This APP predicts the Iris Flower Type
             """)
    st.sidebar.header("User Input Parameters")
    
    df = user_input_features()
    
    st.subheader("User Input Parameters")
    st.write(df)
    
    prediction = iris_prediction(df)
    
    st.subheader("Iris Type Predicted")
    st.write(prediction)
    
    
if __name__ == '__main__':
    main()
