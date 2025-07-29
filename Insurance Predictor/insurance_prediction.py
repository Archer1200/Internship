import pickle
import streamlit as st
model1=pickle.load(open('insurance predict.pkl','rb'))

def predict():
    st.title('insurance prediction')
    age=st.number_input('Age')
    pred=st.button('predict')
    
    if pred:
        x=model1.predict([[age]])
        st.write(x[0])


predict()