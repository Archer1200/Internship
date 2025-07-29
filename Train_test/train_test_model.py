import pickle 
import streamlit as st
model2=pickle.load(open('train_test.pkl','rb'))

def predict():
    st.title('Train Test Model  ')
    test=st.number_input('Enter the value')
    pred=st.button('predict')
    
    if pred:
        x=model2.predict([[test]])
        st.write(x[0])
predict()